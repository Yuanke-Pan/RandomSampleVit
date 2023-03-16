import torch
import torch.nn.functional as F
import numpy as np
import timm

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return torch.tanh(self.fc2(x))


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1).to(torch.float) # 拼接状态和动作
        x = self.activation(self.fc1(cat))
        x = self.activation(self.fc2(x))
        return self.fc_out(x)

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, hidden_ratio, action_dim, sigma, actor_lr, critic_lr, tau, device):
        self.feature_extract = timm.create_model('efficientnet_es', pretrained=False, num_classes=0)
        # can give critic a seperate feature extractor
        rand_input = torch.randn((1, 3, 224, 224))
        rand_output = self.feature_extract(rand_input)
        state_dim = rand_output.shape[1]
        
        self.feature_extract = self.feature_extract.to(device)
        self.actor = PolicyNet(state_dim, int(state_dim * hidden_ratio), action_dim).to(device)
        self.critic = QValueNet(state_dim, int(state_dim * hidden_ratio), action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, int(state_dim * hidden_ratio), action_dim).to(device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.sigma = sigma  
        self.tau = tau
        self.action_dim = action_dim
        self.device = device

    def take_noise_action(self, image):
        image = image.to(self.device)
        state = self.feature_extract(image)
        action = self.actor(state).cpu()
        
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def take_action(self, image):
        image = torch.tensor(image, dtype=torch.float).to(self.device)
        state = self.feature_extract(image)
        action = self.actor(state)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, image, action, reward):
        image = image.to(self.device)
        actions = action.to(self.device)
        rewards = reward.to(self.device)

        states = self.feature_extract(image)
        critic_state = states.clone().detach()

        q_targets = rewards.to(torch.float)
        critic_loss = torch.mean(F.mse_loss(self.critic(critic_state, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)