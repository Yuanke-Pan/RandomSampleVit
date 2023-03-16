import torch
from torch import nn 

from models.vit import SampleVisionTransformer
from models.SampleAgent import DDPG

class SampleLayer(nn.Module):
    def __init__(self, num_samples, patch_size, image_size, device) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.image_size = image_size
        self.device = device
    
    def forward(self, image, mu, beta):
        batch_size = image.shape[0]
        x_beta = beta[:, 0].unsqueeze(-1).expand(-1, self.num_samples).to(self.device)
        x_mu = mu[:, 0].unsqueeze(-1).expand(-1, self.num_samples).to(self.device)
        y_beta = beta[:, 1].unsqueeze(-1).expand(-1, self.num_samples).to(self.device)
        y_mu = mu[:, 1].unsqueeze(-1).expand(-1, self.num_samples).to(self.device)
        x = torch.clamp(x_beta * torch.randn((batch_size, self.num_samples)).to(self.device) + x_mu, min = 0.0, max = 1)
        y = torch.clamp(y_beta * torch.randn((batch_size, self.num_samples)).to(self.device) + y_mu, min = 0.0, max = 1)
        x = torch.clamp(torch.round(x * self.image_size), min = 0.0, max = self.image_size - self.patch_size)
        y = torch.clamp(torch.round(y * self.image_size), min = 0.0, max = self.image_size - self.patch_size)
        z = torch.stack([x, y], dim=2)
        z, _ = torch.sort(z, dim=1)
        z = z.long()
        all_patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        img_ind = torch.arange(batch_size)
        img_ind = img_ind.unsqueeze(-1).repeat(1, self.num_samples)
        start_x = z[:, :, 0]
        start_y = z[:, :, 1]
        selected_patches = all_patches[img_ind, :, start_x, start_y, :, :]
        selected_patches = selected_patches.reshape(batch_size, 3, self.num_samples * self.patch_size, self.patch_size)
        #Could add unique function here
        return selected_patches

class SpVit(nn.Module):
    def __init__(self, image_size, num_sample, patch_size, actor_lr, critic_lr, device, num_classes = 1000, global_pool='token', embed_dim=768, depth=12, num_head=12, mlp_ratio=4.) -> None:
        super().__init__()
        self.RL_agent = DDPG(hidden_ratio=2., action_dim=4, sigma=0.3, actor_lr=actor_lr, critic_lr=critic_lr, tau=0.9, device=device)
        self.SampleLayer = SampleLayer(num_samples=num_sample, patch_size=patch_size, image_size=image_size, device=device)
        self.vit = SampleVisionTransformer(num_sample=num_sample, patch_size=patch_size, num_classes=num_classes, global_pool=global_pool, embed_dim=embed_dim,
                                           depth=depth, num_head=num_head, mlp_ratio=mlp_ratio)

    def forward(self, image):
        if self.training:
            with torch.no_grad():
                distribution_parameter = self.RL_agent.take_noise_action(image)
        else:
            with torch.no_grad():
                distribution_parameter = self.RL_agent.take_action(image)
        mu, beta = distribution_parameter[:, :2], distribution_parameter[:, 2:]
        x = self.SampleLayer(image, mu, beta)
        x = self.vit(x)
        return x, distribution_parameter
