import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MPC planning algorithm
class MPC():
    def __init__(self, action_dim):
        self.horizon = 12
        self.opt_iters = 4 
        self.num_candidates = 100 
        self.num_best_candidates = 10
        self.action_dim = action_dim
        
    def get_action(self, det_state, stoc_state, rssm, reward_model):
        action_mean = torch.zeros(self.horizon, self.action_dim)
        action_dev = torch.ones(self.horizon, self.action_dim)
        action_dist = torch.distributions.Normal(action_mean, action_dev)
        for _ in range(self.opt_iters):
            candidate_actions = action_dist.sample((self.num_candidates,))
            candidate_rewards = []
            for c in range(self.num_candidates):
                # Reset states
                planning_det_state = det_state
                planning_stoc_state = stoc_state
                horizon_reward = 0
                for t in range(self.horizon):
                    with torch.no_grad():
                        planning_det_state = rssm.drnn(planning_det_state.to(device), planning_stoc_state.to(device), candidate_actions[c][t].to(device))
                        planning_stoc_state, _, _ = rssm.ssm_prior(planning_det_state.to(device))
                        planning_stoc_state = planning_stoc_state
                        horizon_reward += reward_model(torch.cat((planning_det_state, planning_stoc_state)).to(device))
                candidate_rewards.append(horizon_reward)
            _, best_candidates = torch.topk(torch.as_tensor(candidate_rewards), self.num_best_candidates)
            best_actions = candidate_actions[best_candidates]
            action_dev, action_mean = torch.std_mean(best_actions, dim=0)
            action_dist = torch.distributions.Normal(action_mean, action_dev)
        # Return best action for t=0 + noise
        return action_mean[0] + torch.randn_like(action_mean[0]) * 0.3 