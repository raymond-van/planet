import torch
from torch import nn
from torch.nn import functional as F

# Convolutional encoder
# Extracts 1024-dimensional features from observations
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

    def forward(self, obs):
        code = F.relu(self.conv1(obs))
        code = F.relu(self.conv2(code))
        code = F.relu(self.conv3(code))
        code = F.relu(self.conv4(code))
        code = code.view(-1, 1024)
        return code
    
# Transpose conv decoder (Observation model)
# Reconstruct observation from determinstic + stochastic state
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200 + 30, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
 
    def forward(self, det_state, stoc_state):
        rec_obs = F.relu(self.fc1(torch.cat((det_state, stoc_state), dim=1)))
        rec_obs = F.relu(self.deconv1(rec_obs))
        rec_obs = F.relu(self.deconv2(rec_obs))
        rec_obs = F.relu(self.deconv3(rec_obs))
        rec_obs = self.deconv4(rec_obs)
        return rec_obs
    
# Predicts reward from deterministic + stochastic (posterior) state
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200 + 30,200)
        self.fc2 = nn.Linear(200, 1)
        
    def forward(self, det_stoc_state):
        reward = F.relu(self.fc1(det_stoc_state))
        return self.fc2(reward)
    
# Deterministic + stochastic state model
class RSSM(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(30 + action_dim, 200)
        self.gru = nn.GRUCell(200, 200)
        self.fc2 = nn.Linear(200, 30 + 30)
        self.fc3 = nn.Linear(200 + 1024, 30 + 30)
    def forward(self, prev_det, prev_stoc, action, obs_feat=None):
        prev_state_action = F.relu(self.fc1(torch.cat((prev_stoc, action))))
        det_state = self.gru(prev_state_action, prev_det)
        prior_state_mean_dev = torch.split(F.relu(self.fc2(det_state)).unsqueeze(dim=0), 30, dim=1)
        prior_state_mean = prior_state_mean_dev[0]
        prior_state_dev = prior_state_mean_dev[1]
        prior_state = prior_state_mean + prior_state_dev * torch.randn_like(prior_state_mean)
        # return deterministic state, prior stochastic state, posterior stochatic state given obs
        if obs_feat != None:
            post_state_mean_dev = torch.split(F.relu(self.fc3(torch.cat((det_state, obs_feat.squeeze())))).unsqueeze(dim=0), 30, dim=1)
            post_state_mean = post_state_mean_dev[0]
            post_state_dev = post_state_mean_dev[1]
            post_state = post_state_mean + post_state_dev * torch.randn_like(post_state_mean)
            return det_state, prior_state, prior_state_mean, prior_state_dev, post_state, post_state_mean, post_state_dev
        # return deterministic state, prior stochastic state
        else:
            return det_state, prior_state, prior_state_mean, prior_state_dev
    
        
# Testing encoder and decoder to see if working as intended
# Not part of final implementation
# class AutoEncoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.enc = encoder
#         self.dec = decoder
    
#     def forward(self, img):
#         code = self.enc(img)
#         return self.dec(code)
    