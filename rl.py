import gym
import logging
import os
import sys
from model import Agent, Net, CNN
from utils import *
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

parser = ArgumentParser()
parser.add_argument("-train_render_mode", type=str, default="rgb_array")
parser.add_argument("-demo_render_mode", type=str, default="rgb_array")
parser.add_argument("-eps", type=float, default=0.1)
parser.add_argument("-greedy", type=int, default=0)
parser.add_argument("-clip_ratio", type=float, default=0.1)
parser.add_argument("-naive_pg", type=int, default=0)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-learning_rate", type=float, default=1e-3)
parser.add_argument("-num_samples", type=int, default=100)
parser.add_argument("-sequence_model", type=int, default=1)
parser.add_argument("-lr_step", type=int, default=20)
parser.add_argument("-lr_decay_rate", type=float, default=0.5)
parser.add_argument("-image_state", type=int, default=0)
parser.add_argument("-k", type=int, default=10)
args = parser.parse_args()

run_time = datetime.now().isoformat(timespec="seconds")
run_time = run_time.replace(
    ":", "-") if sys.platform[:3] == "win" else run_time
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
log_root = f"./runs-{run_time}"
if not os.access(log_root, os.F_OK):
    os.makedirs(log_root)
log_file = f"{log_root}/log.txt"
logfile = logging.FileHandler(log_file)
logger.addHandler(logfile)
cartpole = gym.envs.make("CartPole-v1", render_mode=args.train_render_mode)
demo_env = gym.envs.make("CartPole-v1", render_mode=args.demo_render_mode)

agent_memory_length = 500
if args.sequence_model:
    model = Agent(state_dim=4, num_actions=2, num_layers=2,
                  memory_length=agent_memory_length, interpolate_scale=4, dim_forward=32, max_len=args.num_samples)
else:
    model = Net(4, 2, max_len=args.num_samples)
if args.image_state:
    model = CNN(args.num_samples)
actor = deepcopy(model)
args.sequence_model = isinstance(model, Agent)
actor.load_state_dict(model.state_dict())
actor.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
actor.to(device)
model.to(device)
num_parameters = sum([p.numel() for p in model.parameters()])
for k, v in args.__dict__.items():
    logger.info(f"{k}: {v}")
logger.info(f"number of parameters: {num_parameters}")
logger.info(f"running on: {device}")
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, args.lr_step, gamma=args.lr_decay_rate)
for e in tqdm(range(args.epochs)):
    eps = torch.clip(torch.tensor(
        args.eps ** (e / 100 + 1), device=device), 0, 0.25)
    baseline = generate_memeory(
        actor, cartpole, device, args.num_samples, args.sequence_model, args.image_state, eps=eps, greedy=args.greedy)
    epi_len = [len(s[0]) for s in actor.storage]
    logger.info(f"epoch:{e}; episode lengths: {epi_len}")
    dataset = CartpoleDataset(actor.storage)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate if args.sequence_model else linear_collate)
    k = 1 if args.naive_pg else args.k
    for u in range(k):
        for i, batch in enumerate(dataloader):
            '''
            Shapes:
            States: [B,S,D] for transformer, [B,D] for Linear
            actions: [B,S,N] for transformer, [B,N] for Linear
            rewards:[B,S]
            returns:[B,S]
            '''
            if args.sequence_model:
                train_sequantial(model, actor, batch, optimizer,
                                 device, args.naive_pg, baseline, args.clip_ratio)
            else:
                train_linear(model, actor, batch, optimizer,
                             device, args.naive_pg, baseline, args.clip_ratio, args.image_state)
    lr_scheduler.step()
    tr, images = model.run(demo_env, device)
    if len(images)!=0:
        imageio.mimsave(f"{log_root}/test_run_{e}.gif", images)
    logger.info(
        f"epoch: [{e}]; learning_rate: {lr_scheduler.get_last_lr()} previous baseline: [{baseline}];  total return: [{tr}]")
    actor.load_state_dict(model.state_dict())
    actor.eval()
