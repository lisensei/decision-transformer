import torch
import torch.nn as nn
from collections import deque
import numpy as np
from data import center_crop


class Agent(nn.Module):
    def __init__(self, state_dim, num_actions, num_layers=2, max_len=100, interpolate_scale=4, num_heads=4, dim_forward=32, memory_length=100, batch_first=True) -> None:
        super().__init__()
        self.storage = deque(maxlen=max_len)
        self.num_heads = num_heads
        self.memory_length = memory_length
        self.encoder_dim = interpolate_scale*state_dim
        self.interpolation = nn.Sequential(
            nn.Linear(state_dim, self.encoder_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.encoder_dim)
        )
        self.position_embedding = nn.Embedding(memory_length, self.encoder_dim)
        self.custom_encoder = nn.TransformerEncoderLayer(
            self.encoder_dim, num_heads, dim_forward, batch_first=batch_first)
        self.backbone = nn.TransformerEncoder(self.custom_encoder, num_layers)
        self.fc = nn.Linear(self.encoder_dim, num_actions)

    def forward(self, source):
        if source.dim() != 3:
            raise RuntimeError(f"expected x has 3 dims but got {source.dim()}")
        b, segment_length = source.size(0), source.size(1)
        x = source.reshape(b*segment_length, -1)
        x = self.interpolation(x).reshape(b, segment_length, -1)
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
    def run(self, env, device, greedy=True):
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
                action = torch.argmax(out[0, -1]).item()
            else:
                action = torch.multinomial(
                    torch.softmax(out[0, -1], dim=0), 1).item()
            if action >= env.action_space.n:
                action = env.action_space.sample()
            state, r, done, _, truncated = env.step(action)
            states.append(state)
            env.render()
            total_returns += r
            if total_returns >= 500:
                break
        return total_returns


class Net(nn.Module):
    def __init__(self, state_size, action_size, max_len=200):
        super().__init__()
        self.storage = deque(maxlen=max_len)
        self.layers = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, action_size)
        )

    def forward(self, x):
        return self.layers(x)

    @torch.no_grad()
    def run(self, env, device, greedy=True):
        self.eval()
        state, _ = env.reset()
        done = False
        truncated = False
        total_returns = 0
        while not done and not truncated:

            out = self.forward(torch.tensor(np.array(state), device=device).reshape(
                -1, env.observation_space.shape[0]))
            if greedy:
                action = torch.argmax(out, dim=1)
            else:
                action = torch.multinomial(torch.softmax(out, dim=1), 1)
            state, r, done, _, truncated = env.step(action.item())
            env.render()
            total_returns += r
            if total_returns >= 500:
                break
        return total_returns


class CNN(nn.Module):
    def __init__(self, maxlen) -> None:
        self.storage = deque(maxlen=maxlen)
        super().__init__()
        self.con1 = nn.Conv2d(3, 32, 7, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.con2 = nn.Conv2d(32, 32, 5, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.con3 = nn.Conv2d(32, 64, 3, 2)
        self.bn3 = nn.BatchNorm2d(64)
        self.con4 = nn.Conv2d(64, 16, 3, 1)
        self.bn4 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(144, 2)

    def forward(self, x):
        x = self.bn1(torch.relu(self.con1(x)))
        x = self.bn2(torch.relu(self.con2(x)))
        x = self.bn3(torch.relu(self.con3(x)))
        x = self.bn4(torch.relu(self.con4(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    @torch.no_grad()
    def run(self, env, device, greedy=True):
        self.eval()
        assert env.render_mode == "rgb_array", "environment needs to be in rgb_array mode"
        ob, _ = env.reset()
        state = env.render()
        states = []
        done = False
        truncated = False
        rewards = 0
        while not done and not truncated:
            states.append(state)
            out = self.forward(torch.tensor(
                np.array(center_crop(ob[0], states[-1])), device=device).permute(2, 0, 1).div(255).unsqueeze(0))
            if greedy:
                action = torch.argmax(out).item()
            else:
                probs = torch.softmax(out, 0)
                action = torch.multinomial(probs, 1).item()
            ob, r, done, _, truncated = env.step(action)
            state = env.render()
            rewards += r
            if rewards >= 500:
                break
        return rewards, states


if __name__ == "__main__":
    state_size = 4
    action_size = 2
    net = Net(state_size, action_size)
    x = torch.randn(size=(2, state_size))
    net(x)
