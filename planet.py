# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# get_ipython().run_line_magic('env', 'MUJOCO_GL=egl')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dm_control import suite
from dm_control.suite.wrappers import pixels
from models import Encoder, Decoder, AutoEncoder
from replay import ExpReplay
from torch import optim
from torch.nn import functional as F
from utils import display_img, display_video, preprocess_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.rcParams['animation.embed_limit'] = 2**128
random_state = np.random.RandomState(0)

# For animations to render inline in jupyter,
# download ffmpeg and set the path below to the location of the ffmpeg executable
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


SEED_EPS = 4
BATCH_SZ = 30
CHUNK_LEN = 20
img_shape = (3, 64, 64)
data = ExpReplay(BATCH_SZ, CHUNK_LEN)

env = suite.load('cheetah', 'run')
env = pixels.Wrapper(env) # only use pixels instead of internal state
act_spec = env.action_spec()


# Generate random seed data
for i in range(SEED_EPS):
    state = env.reset()
    while not state.last():
        action = random_state.uniform(act_spec.minimum, act_spec.maximum, act_spec.shape)
        reward = state.reward
        obs = env.physics.render(camera_id=0, height=200, width=200)
        obs = preprocess_img(obs)
        data.replay.append((obs, action, reward))
        state = env.step(action)

def get_obs_from_data(obs, replay):
    for i in range(len(replay)):
        obs.append(replay[0][0])
        
obs = []
get_obs_from_data(obs, data.replay)

enc = Encoder()
dec = Decoder()
autoencoder = AutoEncoder(enc, dec).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3, eps=1e-4)
loss_fn = torch.nn.MSELoss()
epochs = 15

train_data = obs[:3200]
test_data = obs[3200:]

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=32, shuffle=False, num_workers=4
)


losses = []
for epoch in range(epochs):
    loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(batch)
        train_loss = loss_fn(outputs, batch)
        train_loss.backward()
        optimizer.step()

        loss += train_loss.item()

    loss = loss / len(train_loader)
    losses.append(loss)

    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


plt.plot(losses)


test_batch = next(iter(test_loader))
test_ex = test_batch[0]
display_img(test_ex)

rec_test = autoencoder(test_ex.to(device))
display_img(torch.squeeze(rec_test.cpu()).detach())

