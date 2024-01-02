import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CartpoleDataset(Dataset):
    def __init__(self, episodes):
        super().__init__()
        self.episodes = episodes

    def __getitem__(self, index):
        return self.episodes[index]

    def __len__(self):
        return len(self.episodes)


@torch.no_grad()
def sample_episode(agent, env, device, sequence_model=True, eps=0.1, greedy=False, save_to_agent_memory=True,):
    agent.eval()
    state_dim = env.observation_space.shape[0]
    states = []
    actions = []
    rewards = []
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        states.append(state)
        if not sequence_model:
            x = torch.tensor(np.array(states[-1]), device=device).reshape(
                1, state_dim)
        else:
            x = torch.tensor(np.array(states), device=device).reshape(
                1, -1, state_dim)
        output = agent(x)
        if greedy:
            if not sequence_model:
                a = torch.argmax(output, 1).item()
            else:
                a = torch.argmax(output[0, -1]).item()
        else:
            if not sequence_model:
                probs = torch.softmax(output, 1)
            else:
                probs = torch.softmax(
                    output[0, -1], dim=0)
            if torch.isnan(probs).sum() >= 1:
                a = env.action_space.sample()
            else:
                a = torch.multinomial(probs, 1).item()
        if a >= env.action_space.n:
            a = env.action_space.sample()
        actions.append(a)
        state, r, done, _, truncated = env.step(a)
        rewards.append(r)
        if sum(rewards) > 500:
            break

    states = torch.tensor(np.array(states), device=device)
    actions = torch.tensor(np.array(actions), device=device)
    rewards = torch.tensor(rewards, device=device)
    if save_to_agent_memory:
        agent.storage_capacity.append((states, actions, rewards))
    return states, actions, rewards, sum(rewards)


def generate_memeory(agent, env, device, num_episodes, sequence_model=True, save=True, eps=0.1):
    agent.storage_capacity.clear()
    rewards = []
    for i in range(num_episodes):
        _, _, _, r = sample_episode(
            agent, env, device, sequence_model=sequence_model, eps=eps, save_to_agent_memory=save)
        rewards.append(r)
    return sum(rewards)/len(rewards)


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
    for _, actions, _ in batch:
        episode_length = len(actions)
        max_len = episode_length if max_len < episode_length else max_len
    for states, actions, rewards in batch:
        state_length = len(states)
        step_returns = compute_total_return(rewards)
        if state_length < max_len:
            diff = max_len-state_length
            state_pad_value = torch.empty(
                size=(diff, states.size(1))).fill_(0)
            states = torch.cat([states, state_pad_value])
            action_pad_value = torch.ones(size=(diff,))*2
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
    for states, actions, rewards in batch:
        step_returns = compute_total_return(rewards)
        states = torch.cat([states])
        actions = torch.cat([actions])
        rewards = torch.sum(rewards).expand(states.size(0))
        step_returns = torch.cat([step_returns])
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
        total_returns.append(step_returns)
    return torch.cat(state_list), torch.cat(action_list), torch.cat(reward_list).unsqueeze(1), torch.cat(total_returns).unsqueeze(1)


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
