# PlaNet
Reproducing the [Deep Planning Network (PlaNet) by Hafner et al.](https://arxiv.org/pdf/1811.04551.pdf).
PlaNet is a model-based RL agent that leverages its world model to *plan in latent space*. Planning in the context of RL refers to giving the agent time (or compute) to think about the best course of action to take. Doing this entirely in low-dimensional latent space allows the agent to imagine thousands of different state-action trajectories, selecting the action that corresponds to most reward. The result is 200x sample efficiency gains over comparable model-free methods.

## Dependencies:
- PyTorch
- DeepMind Control Suite
- ffmpeg (to render inline jupyter animations)

## Checklist:
- [x] Generate random seed episodes
- [ ] Implement experience replay
- [ ] Define transition model
- [ ] Define reward model
- [ ] Define state encoder
- [ ] Implement planner
- [ ] Put everything together and train
