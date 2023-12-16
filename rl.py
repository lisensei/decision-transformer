import gym
import torch
import torch.nn as nn
import numpy as np


class Agent(nn.Module):
    def __init__(self, state_dim, num_actions, interpolate_scale=16, num_heads=4, dim_forward=512, memory_length=100, batch_first=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.memory_length = memory_length
        self.encoder_dim = interpolate_scale*state_dim
        self.interpolation = nn.Linear(state_dim, self.encoder_dim)
        self.position_embedding = nn.Embedding(memory_length, self.encoder_dim)
        self.custom_encoder = nn.TransformerEncoderLayer(
            self.encoder_dim, num_heads, dim_forward, batch_first=batch_first)
        self.backbone = nn.TransformerEncoder(self.custom_encoder, 2)
        self.fc = nn.Linear(self.encoder_dim, num_actions+1)

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


@torch.no_grad()
def sample_episode(agent, env, device):
    agent.eval()
    state_dim = env.observation_space.shape[0]
    states = []
    actions = []
    rewards = []
    state, _ = env.reset()
    done = False
    while not done:
        states.append(state)
        x = torch.tensor(np.array(states), device=device).reshape(
            1, -1, state_dim)
        output = agent(x)
        a = torch.argmax(output[:, -1], dim=1).numpy()[0]
        if a >= env.action_space.n:
            a = env.action_space.sample()
        actions.append(a)
        state, r, done, _, _ = env.step(a)
        rewards.append(r)
        pic = env.render()

    states = torch.tensor(np.array(states), device=device)
    actions = torch.tensor(np.array(actions), device=device)
    rewards = torch.tensor(rewards, device=device)
    print(states.shape, actions.shape, rewards.shape)

    return states, actions, rewards


def collate(batch):
    state_list = []
    action_list = []
    reward_list = []
    max_len = -1
    for _, actions, _ in batch:
        episode_length = len(actions)
        max_len = episode_length if max_len < episode_length else max_len
    for states, actions, rewards in batch:
        state_length = len(states)
        if state_length < max_len:
            diff = max_len-state_length
            state_pad_value = torch.zeros(
                size=(diff, states.size(1)))
            states = torch.cat([states, state_pad_value])
            action_pad_value = torch.ones(size=(diff,))*2
            actions = torch.cat([actions, action_pad_value])
            reward_pad_value = torch.zeros(size=(diff,))
            rewards = torch.cat([rewards, reward_pad_value])
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
    return torch.stack(state_list), torch.stack(action_list), torch.stack(reward_list)


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


cartpole = gym.envs.make("CartPole-v1", render_mode="human")

model = Agent(4, 2)
num_parameters = sum([p.numel() for p in model.parameters()])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"number of parameters: {num_parameters}")
