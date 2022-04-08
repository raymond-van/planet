#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# get_ipython().run_line_magic('env', 'MUJOCO_GL=egl')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dm_control import suite
from dm_control.suite.wrappers import pixels
from models import Encoder, Decoder, RewardModel, RSSM
from mpc import MPC
from replay import ExpReplay
from torch import optim
from torch.nn import functional as F
from utils import display_img, display_video, preprocess_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.rcParams['animation.embed_limit'] = 2**128
random_state = np.random.RandomState(0)


# In[2]:


# For animations to render inline in jupyter,
# download ffmpeg and set the path below to the location of the ffmpeg executable
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


# In[3]:


SEED_EPS = 2
TRAIN_EPS = 100
UPDATES = 100
ACTION_REPEAT = 8


# In[4]:


env = suite.load('cartpole', 'swingup')
env = pixels.Wrapper(env) # only use pixels instead of internal state
act_spec = env.action_spec()
action_dim = act_spec.shape[0]

data = ExpReplay()


# In[5]:


# Generate random seed data
total_reward_seed = 0
t = 0
for i in range(SEED_EPS):
    state = env.reset()
    reward = 0
    while not state.last():
        t += 1
        action = random_state.uniform(act_spec.minimum, act_spec.maximum, action_dim)
        reward = state.reward
        if reward:
            total_reward_seed += reward
        frame = env.physics.render(camera_id=0, height=200, width=200)
        frame = preprocess_img(frame)
        data.replay.append((frame, action, reward))
        state = env.step(action)
print("Avg reward per ep: ",total_reward_seed/SEED_EPS)
print("Avg timesteps per ep: ", t/SEED_EPS)


# In[6]:


def extract_from_replay(replay):
    obs = []
    rewards = []
    actions = []
    for i in range(len(replay)):
        obs.append(replay[i][0])
        actions.append(replay[i][1])
        rewards.append(replay[i][2])
    return obs, actions, rewards
obs, actions, rewards = extract_from_replay(data.replay)


# In[7]:


obs[0].shape


# In[8]:


enc = Encoder().to(device)
dec = Decoder().to(device)
reward_model = RewardModel().to(device)
rssm = RSSM(action_dim).to(device)
optimizer = optim.Adam(rssm.parameters(), lr=1e-3, eps=1e-4)

planner = MPC(action_dim)

