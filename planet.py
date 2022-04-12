# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# get_ipython().run_line_magic('env', 'MUJOCO_GL=egl')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torchvision
from dm_control import suite
from dm_control.suite.wrappers import pixels
from models import Encoder, Decoder, RewardModel, RSSM
from mpc import MPC
from replay import ExpReplay
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import display_img, display_video, preprocess_img, save_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.rcParams['animation.embed_limit'] = 2**128
random_state = np.random.RandomState(0)

# For animations to render inline in jupyter,
# download ffmpeg and set the path below to the location of the ffmpeg executable
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

SEED_EPS = 5
TRAIN_EPS = 100
UPDATES = 100
BATCH_SZ = 50
CHUNK_LEN = 50
LOAD_WEIGHTS = 149
DOMAIN_TASK = "cheetah_run"
domain, task = DOMAIN_TASK.split("_")
ACTION_REPEAT = 6

env = suite.load(domain, task, task_kwargs={'random': random_state})
env = pixels.Wrapper(env) # only use pixels instead of internal state
act_spec = env.action_spec()
action_dim = act_spec.shape[0]
data = ExpReplay(BATCH_SZ, CHUNK_LEN, action_dim)


# rand_action_gif = []
# Generate random seed data
rand_rewards = []
total_reward_seed = 0
for i in range(SEED_EPS):
    state = env.reset()
    reward = 0
    ep_reward = 0
    while not state.last():
        action = random_state.uniform(act_spec.minimum, act_spec.maximum, action_dim)
        reward = state.reward
        if reward is None: reward = 0
        ep_reward += reward
        frame = env.physics.render(camera_id=0, height=64, width=64)
        # rand_action_gif.append(frame)
        frame = preprocess_img(frame).to(device)
        data.append(frame, torch.as_tensor(action), torch.as_tensor(reward))
        state = env.step(action)
    rand_rewards.append(ep_reward)
print("Random Action - Avg reward/ep: ",total_reward_seed/SEED_EPS)

enc = Encoder().to(device)
dec = Decoder().to(device)
reward_model = RewardModel().to(device)
rssm = RSSM(action_dim).to(device)
params = list(enc.parameters()) + list(dec.parameters()) + list(reward_model.parameters()) + list(rssm.parameters())
optimizer = optim.Adam(params, lr=1e-3, eps=1e-4)
planner = MPC(action_dim)
 
rewards_list = []
losses_list = []
observations = []
free_nats = torch.tensor(3).to(device)
TRAIN_EPS = 150
# Train for 150 eps
for ep in tqdm(range(LOAD_WEIGHTS+1, TRAIN_EPS)):
    # MODEL FITTING
    total_loss = 0
    for j in range(UPDATES):
        obs, actions, rewards = data.sample_batch()
        state = env.reset()
        obs_loss, reward_loss, kl_div = 0, 0, 0
        det_state = torch.zeros((50,200)).to(device)
        stoc_state = torch.zeros((50,30)).to(device)
        prior_mean, prior_dev, post_mean, post_dev = torch.zeros((50,30)).to(device), torch.zeros((50,30)).to(device),                                                      torch.zeros((50,30)).to(device), torch.zeros((50,30)).to(device)
        for b in range(BATCH_SZ):
            det_state = rssm.drnn(det_state, stoc_state, actions[b])
            prior_state, prior_mean, prior_dev = rssm.ssm_prior(det_state)
            posterior_state, post_dev, post_dev = rssm.ssm_posterior(det_state, enc(obs[b]))
            obs_loss += F.mse_loss(dec(det_state, stoc_state), obs[b])
            reward_loss += F.mse_loss(reward_model(torch.cat((det_state, stoc_state),dim=1)), rewards[b])
            prior_state_dist = torch.distributions.Normal(prior_mean, prior_dev)
            posterior_state_dist = torch.distributions.Normal(post_mean, post_dev)
            kl_div += torch.max(torch.distributions.kl_divergence(prior_state_dist, posterior_state_dist).sum(1), free_nats).mean()
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(params, 1000., norm_type=2)
        loss = obs_loss + reward_loss + kl_div
        loss.backward()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses_list.append(total_loss)
    # DATA COLLECTION
    eps_reward = 0
    vid = []
    state = env.reset()
    frame = env.physics.render(camera_id=0, height=64, width=64)
    with torch.no_grad():
        det_state = torch.zeros(200).to(device)
        stoc_state = torch.zeros(30).to(device)
        action = torch.zeros(action_dim).to(device)
        frame = preprocess_img(env.physics.render(camera_id=0, height=200, width=200)).to(device)
        while not state.last():
            det_state = rssm.drnn(det_state, stoc_state, action.to(device))
            stoc_state, _, _ = rssm.ssm_posterior(det_state, enc(frame))
            stoc_state = stoc_state.squeeze()
            action = planner.get_action(det_state.to(device), stoc_state.to(device), rssm, reward_model)
            for _ in range(ACTION_REPEAT):
                if state.last(): break
                state = env.step(action)
                eps_reward += state.reward
                frame = env.physics.render(camera_id=0, height=64, width=64)
                vid.append(frame)
            frame = preprocess_img(frame).to(device)
            data.append(frame, action, state.reward)
        rewards_list.append(eps_reward)
        observations.append(vid)
    if (ep % 10 == 0):
        torch.save({'rssm': rssm.state_dict(), 'decoder': dec.state_dict(),                     'reward': reward_model.state_dict(), 'encoder': enc.state_dict()},                      os.path.join('weights', '{task}_{eps}.pth'.format(task=DOMAIN_TASK, eps=ep)))
        anim = display_video(vid, return_anim=True)
        anim.save('gifs/{task}_{eps}.gif'.format(task=DOMAIN_TASK, eps=ep), writer='imagemagick', fps=30)
    print("Episode: {}/{}, Loss = {:.2f}, Reward = {:.2f}".format(ep, TRAIN_EPS, total_loss, eps_reward))

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(rewards_list)

plt.xlabel("Episode")
plt.ylabel("Loss")
plt.plot(losses_list)

