import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


class CartpoleDataset(Dataset):
    def __init__(self, episodes):
        super().__init__()
        self.episodes = episodes

    def __getitem__(self, index):
        return self.episodes[index]

    def __len__(self):
        return len(self.episodes)


@torch.no_grad()
def sample_episode(agent, env, device, sequence_model=True, image_state=False,  greedy=False, eps=0.1):
    '''
    returns:
    observation: Tensor
    actions: Tensor
    rewards: Tensor
    episode return: int
    images: PIL image
    '''
    agent.eval()
    state_dim = env.observation_space.shape[0]
    states = []
    observations = []
    actions = []
    rewards = []
    images = []
    observation, _ = env.reset()
    if image_state:
        assert env.render_mode == "rgb_array", f"env needs to be in rgb_array render mode"
        image = env.render()
        image = center_crop(observation[0], image)
        state = image
    else:
        state = observation
    done = False
    truncated = False
    while not done and not truncated:
        states.append(state)
        if not sequence_model:
            if image_state:
                x = torch.tensor(
                    np.array(states[-1]), device=device).permute(2, 0, 1).unsqueeze(0).div(255)
            else:
                x = torch.tensor(np.array(states[-1]), device=device).reshape(
                    1, state_dim)
        else:
            x = torch.tensor(np.array(states), device=device).reshape(
                1, -1, state_dim)
        output = agent(x)
        if greedy:
            if np.random.rand() > eps:
                if not sequence_model:
                    a = torch.argmax(output, 1).item()
                else:
                    a = torch.argmax(output[0, -1]).item()
            else:
                a = env.action_space.sample()
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
        actions.append(a)
        if image_state:
            image = env.render()
            image = center_crop(observation[0], image)
            state = image
            images.append(image)
        else:
            state = observation
        observations.append(observation)
        observation, r, done, _, truncated = env.step(a)
        rewards.append(r)
        if sum(rewards) > 500:
            break
    assert len(states) == len(
        observations), f"states len ={len(states)} observations len== {len(observations)}"
    observations = torch.tensor(np.array(observations), device=device)
    actions = torch.tensor(np.array(actions), device=device)
    rewards = torch.tensor(rewards, device=device)
    if image_state:
        return observations, actions, rewards, sum(rewards), images
    else:
        return observations, actions, rewards, sum(rewards), None


def generate_memeory(agent, env, device, num_episodes, sequence_model=True, image_state=False, greedy=True, eps=0.1):
    agent.storage.clear()
    epi_rewards = []
    for i in range(num_episodes):
        states, actions, rewards, r, images = sample_episode(
            agent, env, device, sequence_model=sequence_model, image_state=image_state, greedy=greedy, eps=eps)
        agent.storage.append((states, actions, rewards, images))
        epi_rewards.append(r)
    return sum(epi_rewards) / len(epi_rewards)


def center_crop(position, image: Image.Image, box_size=[160, 160]):
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    _, box_width = box_size
    width, height = image.size
    half_width = box_width//2
    image_center = width//2
    center = position/2.4*image_center+image_center
    xmin, xmax = center-half_width, center+half_width
    ymin = (height-box_size[1])*2//3
    ymax = ymin+box_size[1]
    if xmin < 0:
        xmin = 0
        xmax = box_width
    if xmax > width:
        xmin = width-box_width
        xmax = width
    # print(
    #    f"image center: {image_center};x:{position},center={center};xmin={xmin};xmax={xmax}")
    box = [xmin, ymin, xmax, ymax]
    image = image.crop(box)
    return image
