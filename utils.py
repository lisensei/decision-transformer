from data import *
import torchvision.transforms.functional as TF
import imageio


def train_linear(model, actor, batch, optimizer, device, naive_pg, baseline, clip_ratio, image_state=False):
    model.train()
    actor.eval()
    states, actions, rewards, returns, images = batch
    if image_state:
        states = images
    states = states.to(device)
    actions = actions.to(device).to(torch.int64)
    rewards = rewards.to(device)
    returns = returns.to(device)
    out = model(states)
    index = actions.unsqueeze(1)
    out = torch.softmax(out, dim=1)
    with torch.no_grad():
        prime = actor(states)
        prime = torch.softmax(prime, 1)
        prime = torch.gather(
            prime, 1, index)
    target = torch.gather(out, 1, index)
    ratio = target/prime
    advantage = returns - baseline
    advantage = torch.min(
        ratio*advantage, torch.clip(ratio, 1-clip_ratio, 1+clip_ratio)*advantage)
    loss = -torch.mean(advantage)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if naive_pg:
        actor.load_state_dict(model.state_dict())
        actor.eval()


def train_sequantial(model, actor, batch, optimizer, device, naive_pg, baseline, clip_ratio):
    model.train()
    actor.eval()
    states, actions, rewards, returns = batch
    states = states.to(device)
    actions = actions.to(device).to(torch.int64)
    rewards = rewards.to(device)
    returns = returns.to(device)
    out = model(states)
    mask = actions != -1
    index = actions[mask].unsqueeze(1)
    returns = returns[mask]
    out = out[mask]
    out = torch.softmax(out, dim=1)
    with torch.no_grad():
        prime = actor(states)[mask]
        prime = torch.softmax(prime, 1)
        prime = torch.gather(
            prime, 1, index)
    target = torch.gather(out, 1, index)
    ratio = (target/prime).squeeze(1)
    assert not ratio.dim == 1, Exception(f"dim error")
    advantage = returns - baseline
    advantage = torch.min(
        ratio*advantage, torch.clip(ratio, 1-clip_ratio, 1+clip_ratio)*advantage)
    loss = -torch.mean(advantage)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if naive_pg:
        actor.load_state_dict(model.state_dict())
        actor.eval()


def resize_image(image, size=[300, 200]):
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize(size)
    return image


def compute_total_return(returns, gamma=1):
    episodo_length = len(returns)
    step_returns = []
    for i in range(episodo_length):
        powers = torch.pow(gamma, torch.arange(
            episodo_length-i, device=returns.device).reshape(-1))
        step_return = sum(returns[i:]*powers)
        step_returns.append(step_return)
    return torch.stack(step_returns)


def collate(batch):
    state_list = []
    action_list = []
    reward_list = []
    total_returns = []
    max_len = -1
    for _, actions, _, _ in batch:
        episode_length = len(actions)
        max_len = episode_length if max_len < episode_length else max_len
    for states, actions, rewards, _ in batch:
        state_length = len(states)
        step_returns = compute_total_return(rewards)
        if state_length < max_len:
            diff = max_len-state_length
            state_pad_value = torch.empty(
                size=(diff, states.size(1))).fill_(0)
            states = torch.cat([states, state_pad_value])
            action_pad_value = torch.ones(size=(diff,))*-1
            actions = torch.cat([actions, action_pad_value])
            reward_pad_value = torch.zeros(size=(diff,))
            rewards = torch.cat([rewards, reward_pad_value])
            step_returns = torch.cat([step_returns, reward_pad_value])
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
        total_returns.append(step_returns)
    return torch.stack(state_list), torch.stack(action_list), torch.stack(reward_list), torch.stack(total_returns)


def linear_collate(batch):
    state_list = []
    action_list = []
    reward_list = []
    total_returns = []
    image_list = []
    for states, actions, rewards, images in batch:
        step_returns = compute_total_return(rewards)
        states = torch.cat([states])
        actions = torch.cat([actions])
        rewards = torch.sum(rewards).expand(states.size(0))
        if images !=None:
            images = torch.tensor(np.array(images)).permute(0, 3, 1, 2).div(255)
        step_returns = torch.cat([step_returns])
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
        total_returns.append(step_returns)
        image_list.append(images)
    if images!=None:
        return torch.cat(state_list), torch.cat(action_list), torch.cat(reward_list).unsqueeze(1), torch.cat(total_returns).unsqueeze(1), torch.cat(image_list)
    else:
        return torch.cat(state_list), torch.cat(action_list), torch.cat(reward_list).unsqueeze(1), torch.cat(total_returns).unsqueeze(1),None

def test_padding(model, env, device):
    s1 = sample_episode(model, env, device)
    s2 = sample_episode(model, env, device)
    states, actions, rewards = collate((s1, s2))
    model(states)
    print("state============")
    print(states[0])
    print(states[1])
    print("\naction=======")
    print(actions[0])
    print(actions[1])
    print("\nreward=======")
    print(rewards[0])
    print(rewards[1])


if __name__ == "__main__":
    import os
    path = "C:/Users/sensei/AppData/Local/Programs/Python/Python39/python.exe"
    os.execl(path, "-m", "rl.py")
