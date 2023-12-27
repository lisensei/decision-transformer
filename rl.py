import gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import deque


class Agent(nn.Module):
    def __init__(self, state_dim, num_actions, num_layers=2, max_len=100, interpolate_scale=4, num_heads=4, dim_forward=64, memory_length=100, batch_first=True) -> None:
        super().__init__()
        self.storage_capacity = deque(maxlen=max_len)
        self.num_heads = num_heads
        self.memory_length = memory_length
        self.encoder_dim = interpolate_scale*state_dim
        self.interpolation = nn.Linear(state_dim, self.encoder_dim)
        self.position_embedding = nn.Embedding(memory_length, self.encoder_dim)
        self.custom_encoder = nn.TransformerEncoderLayer(
            self.encoder_dim, num_heads, dim_forward, batch_first=batch_first)
        self.backbone = nn.TransformerEncoder(self.custom_encoder, num_layers)
        self.fc = nn.Linear(self.encoder_dim, num_actions)
    '''
    def forward(self, source):
        if source.dim() != 3:
            raise RuntimeError(f"expected x has 3 dims but got {source.dim()}")
        b, length = source.size(0), source.size(1)
        num_segments = length//self.memory_length
        remainder = length % self.memory_length
        interval = num_segments+1 if remainder != 0 else num_segments
        output = []
        for i in range(interval):
            if i == interval-1:
                x = source[:, -self.memory_length:, :]
            else:
                start = self.memory_length*i
                end = (i+1)*self.memory_length
                x = source[:, start:end, :]
            segment_length = x.size(1)
            x = x.reshape(b*segment_length, -1)
            x = self.interpolation(x).reshape(b, segment_length, -1)
            x = torch.relu(x)
            pe = self.position_embedding(
                torch.arange(segment_length)).unsqueeze(0).expand(b, -1, -1)
            final_embeddding = pe+x
            mask = nn.Transformer.generate_square_subsequent_mask(
                segment_length)
            mask = mask.unsqueeze(0).expand(b*self.num_heads, -1, -1)
            key_pad_mask = torch.sum(x, dim=2) == 0
            x = self.backbone.forward(
                final_embeddding, mask=mask, src_key_padding_mask=key_pad_mask)
            x = self.fc(x)
            x = x[:, -remainder:, :] if i == interval - \
                1 and remainder != 0 else x
            output.append(x)
        output = torch.cat(output, dim=1)
        return output
    '''

    def forward(self, source):
        if source.dim() != 3:
            raise RuntimeError(f"expected x has 3 dims but got {source.dim()}")
        b, segment_length = source.size(0), source.size(1)
        x = source.reshape(b*segment_length, -1)
        x = self.interpolation(x).reshape(b, segment_length, -1)
        x = torch.relu(x)
        pe = self.position_embedding(
            torch.arange(segment_length)).unsqueeze(0).expand(b, -1, -1)
        final_embeddding = pe+x
        mask = nn.Transformer.generate_square_subsequent_mask(
            segment_length)
        mask = mask.unsqueeze(0).expand(b*self.num_heads, -1, -1)
        key_pad_mask = torch.sum(source, dim=2) == 0
        x = self.backbone.forward(
            final_embeddding, mask=mask, src_key_padding_mask=key_pad_mask)
        x = self.fc(x)
        return x

    @torch.no_grad()
    def run(self, env, device,greedy=True):
        self.eval()
        state, _ = env.reset()
        done = False
        truncated = False
        states = [state]
        total_returns = 0
        while not done and not truncated:

            out = self.forward(torch.tensor(np.array(states), device=device).reshape(
                1, -1, env.observation_space.shape[0]))
            if greedy:
                action = torch.argmax(out[0,-1])
            else:
                action=torch.multinomial(torch.softmax(out[0,-1],dim=0),1)
            if action >= env.action_space.n:
                action = env.action_space.sample()
            state, r, done, _, truncated = env.step(action.item())
            states.append(state)
            env.render()
            total_returns += r
        return total_returns


class CartpoleDataset(Dataset):
    def __init__(self, episodes):
        super().__init__()
        self.episodes = episodes

    def __getitem__(self, index):
        return self.episodes[index]

    def __len__(self):
        return len(self.episodes)


@torch.no_grad()
def sample_episode(agent, env, device, save_to_agent_memory=True,greedy=False):
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
        x = torch.tensor(np.array(states), device=device).reshape(
            1, -1, state_dim)
        output = agent(x)
        if greedy:
            a=torch.argmax(output[0,-1]).item()
        else:
            a=torch.multinomial(torch.softmax(output[0,-1],dim=0),1).item()
        if a >= env.action_space.n:
            a = env.action_space.sample()
        actions.append(a)
        state, r, done, _, truncated = env.step(a)
        rewards.append(r)

    states = torch.tensor(np.array(states), device=device)
    actions = torch.tensor(np.array(actions), device=device)
    rewards = torch.tensor(rewards, device=device)
    if save_to_agent_memory:
        agent.storage_capacity.append((states, actions, rewards))
    return states, actions, rewards, sum(rewards)


def generate_memeory(agent, env, device, num_episodes, save=True):
    agent.storage_capacity.clear()
    rewards = []
    for i in range(num_episodes):
        _, _, _, r = sample_episode(agent, env, device, save)
        rewards.append(r)
    return sum(rewards)/len(rewards)


def compute_total_return(returns, gamma=1):
    episodo_length = len(returns)
    step_returns = []
    for i in range(episodo_length):
        powers = torch.pow(gamma, torch.arange(episodo_length-i).reshape(-1))
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


cartpole = gym.envs.make("CartPole-v1", render_mode="rgb_array")
demo_env = gym.envs.make("CartPole-v1", render_mode="human")
num_samples = 320
agent_memory_length = 500
model = Agent(state_dim=4, num_actions=2, num_layers=2,
              memory_length=agent_memory_length, interpolate_scale=4, dim_forward=32,max_len=num_samples)
num_parameters = sum([p.numel() for p in model.parameters()])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"number of parameters: {num_parameters}")
epochs = 20
batch_size = 1
loss_function = nn.CrossEntropyLoss(ignore_index=2, reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for e in range(epochs):
    baseline=generate_memeory(model, cartpole, device, num_samples)
    dataset = CartpoleDataset(model.storage_capacity)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate)
    model.train()
    losses=0
    for i, (states, actions, rewards, returns) in enumerate(dataloader):
        factor = torch.sum(rewards, dim=1).unsqueeze(
            1).expand(-1, states.size(1))-baseline
        
        out = model(states)
        loss = loss_function(out.permute(0, 2, 1), actions.to(torch.int64))
        losses += torch.sum(loss*factor)
        if (i%64==0 or i==num_samples-1) and i!=0:
            optimizer.zero_grad()
            losses=-losses
            losses.backward()
            optimizer.step()
            losses=0
    tr = model.run(demo_env, device)
    print(f"previous baseline: {baseline};  total return: {tr}")
