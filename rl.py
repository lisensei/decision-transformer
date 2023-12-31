import gym
from model import Agent, Net
from utils import *

eps = 0.1
cartpole = gym.envs.make("CartPole-v1", render_mode="rgb_array")
demo_env = gym.envs.make("CartPole-v1", render_mode="human")
num_samples = 320
agent_memory_length = 500
# model = Agent(state_dim=4, num_actions=2, num_layers=2,
#              memory_length=agent_memory_length, interpolate_scale=4, dim_forward=32,max_len=num_samples)
model = Net(4, 2, max_len=num_samples)
num_parameters = sum([p.numel() for p in model.parameters()])
actor = Net(4, 2)
actor.load_state_dict(model.state_dict())
actor.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"number of parameters: {num_parameters}")
epochs = 20
batch_size = 1
loss_function = nn.CrossEntropyLoss(ignore_index=2, reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for e in range(epochs):
    baseline = generate_memeory(actor, cartpole, device, num_samples)
    dataset = CartpoleDataset(actor.storage_capacity)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=linear_collate)
    model.train()
    for i, (states, actions, rewards, returns) in enumerate(dataloader):
        '''
        Shapes:
        States: [B,S,D] for transformer, [B,D] for Linear
        actions: [B,S,N] for transformer, [B,N] for Linear
        rewards:[B,S]
        returns:[B,S]
        '''

        out = model(states)
        out = torch.softmax(out, dim=1)
        index = actions.to(torch.int64).unsqueeze(1)
        with torch.no_grad():
            prime = actor(states)
            prime = torch.softmax(prime, 1)
            prime = torch.gather(
                prime, 0, index)
        target = torch.gather(out, 0, index)
        ratio = target/prime
        advantage = returns-baseline
        advantage = torch.min(
            ratio*advantage, torch.clip(ratio, 1-eps, 1+eps)*advantage)
        loss = -torch.mean(ratio*advantage)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tr = model.run(demo_env, device)
    print(f"previous baseline: {baseline};  total return: {tr}")
    actor.load_state_dict(model.state_dict())
