import torch
from torch import nn
from torch.nn import functional as F

# Convolutional encoder
# Extracts 1024-dimensional features from observations
class Encoder(nn.Module):
    def __init__(self, feature_sz=1024):
        super().__init__()
        self.feature_sz = feature_sz
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

    def forward(self, obs):
        code = F.relu(self.conv1(obs))
        code = F.relu(self.conv2(code))
        code = F.relu(self.conv3(code))
        code = F.relu(self.conv4(code))
        code = code.view(-1, self.feature_sz)
        return code
    
# Transpose conv decoder (Observation model)
# Reconstruct observation from determinstic + stochastic state
class Decoder(nn.Module):
    def __init__(self, det_sz=200, stoc_sz=30, feature_sz=1024):
        super().__init__()
        self.fc1 = nn.Linear(det_sz + stoc_sz, feature_sz)
        self.deconv1 = nn.ConvTranspose2d(feature_sz, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
 
    def forward(self, det_state, stoc_state):
        if det_state.dim() > 1:
            dim = 1
        else:
            dim = 0
        rec_obs = F.relu(self.fc1(torch.cat((det_state, stoc_state), dim=dim))).unsqueeze(dim=det_state.dim()).unsqueeze(dim=det_state.dim())
        rec_obs = F.relu(self.deconv1(rec_obs))
        rec_obs = F.relu(self.deconv2(rec_obs))
        rec_obs = F.relu(self.deconv3(rec_obs))
        rec_obs = self.deconv4(rec_obs)
        return rec_obs
    
# Predicts reward from deterministic + stochastic (posterior) state
class RewardModel(nn.Module):
    def __init__(self, det_sz=200, stoc_sz=30):
        super().__init__()
        self.fc1 = nn.Linear(det_sz + stoc_sz, det_sz)
        self.fc2 = nn.Linear(det_sz, 1)
        
    def forward(self, det_stoc_state):
        reward = F.relu(self.fc1(det_stoc_state))
        return self.fc2(reward).squeeze()
    
# Deterministic + stochastic state model
class RSSM(nn.Module):
    def __init__(self, action_dim, det_sz=200, stoc_sz=30, feature_sz=1024):
        super().__init__()
        self.rnn = RNN(det_sz, stoc_sz, action_dim)
        self.ssm = SSM(det_sz, stoc_sz, feature_sz)
        
    def drnn(self, prev_det, prev_stoc, action):
        det_state = self.rnn(prev_det, prev_stoc, action)
        return det_state
    
    def ssm_prior(self, det_state):
        prior_state, prior_state_mean, prior_state_dev = self.ssm(det_state)
        return prior_state, prior_state_mean, prior_state_dev
   
    def ssm_posterior(self, det_state, obs_feat):
        post_state, post_state_mean, post_state_dev = self.ssm(det_state, obs_feat)
        return post_state, post_state_mean, post_state_dev
    
# Deterministic state model
class RNN(nn.Module):
    def __init__(self, det_sz, stoc_sz, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(stoc_sz + action_dim, det_sz)
        self.gru = nn.GRUCell(det_sz, det_sz)
    
    def forward(self, prev_det, prev_stoc, action):
        if action.dim() > 1:
            dim = 1
        else:
            dim = 0
        prev_state_action = F.relu(self.fc1(torch.cat((prev_stoc, action), dim=dim)))
        det_state = self.gru(prev_state_action, prev_det)
        return det_state

# Stochastic state model
class SSM(nn.Module):
    def __init__(self, det_sz, stoc_sz, feature_sz):
        super().__init__()
        self.stoc_sz = stoc_sz
        self.fc1 = nn.Linear(det_sz, stoc_sz + stoc_sz)
        self.fc2 = nn.Linear(det_sz + feature_sz, stoc_sz + stoc_sz)
        
    def forward(self, det_state, obs_feat=None):
        if det_state.dim() > 1:
            dim = 1
        else:
            dim = 0
        if obs_feat == None:
            prior_state_mean_dev = torch.split(F.relu(self.fc1(det_state)), self.stoc_sz, dim=dim)
            prior_state_mean = prior_state_mean_dev[0]
            prior_state_dev = prior_state_mean_dev[1] + 1e-6 # stddev must be positive
            prior_state = prior_state_mean + prior_state_dev * torch.randn_like(prior_state_mean)
            return prior_state, prior_state_mean, prior_state_dev
        else:  
            post_state_mean_dev = torch.split(F.relu(self.fc2(torch.cat((det_state, obs_feat.squeeze()),dim=dim))), self.stoc_sz, dim=dim)
            post_state_mean = post_state_mean_dev[0]
            post_state_dev = post_state_mean_dev[1] + 1e-6  # stddev must be positive
            post_state = post_state_mean + post_state_dev * torch.randn_like(post_state_mean)
            return post_state, post_state_mean, post_state_dev
    
    
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
    