import torch
from torch import nn
from torch.nn import functional as F

# Convolutional encoder
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
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, code):
        rec_obs = code.view(-1, 1024, 1, 1)
        rec_obs = F.relu(self.deconv1(rec_obs))
        rec_obs = F.relu(self.deconv2(rec_obs))
        rec_obs = F.relu(self.deconv3(rec_obs))
        rec_obs = self.deconv4(rec_obs)
        return rec_obs
    
# Testing encoder and decoder to see if working as intended
# Not part of final implementation
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder
    
    def forward(self, img):
        code = self.enc(img)
        return self.dec(code)