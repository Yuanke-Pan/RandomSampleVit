import torch
from utils.rl_util import ReplayBuffer
from models.SpVit import SpVit
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# import model config and training config
from config.SpVit_model import SpVit_base
from config.training_config import CIFAR10_SpVit

from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch.utils.data import DataLoader

from training_schedual.SpVit_training import train

from matplotlib import pyplot as plt

import pandas as pd

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

model_name = "SpVit_base"
dataset_name = "CIFAR10"
training_config_name = dataset_name + "_" + model_name.split('_')[0]
device = torch.device('cuda')

epochs = 100

def get_train_parameter(config_name):
    if config_name == "CIFAR10_SpVit":
        cfg = CIFAR10_SpVit
    else:
        assert("we didn't get training parameter")
    return cfg

def get_model_parameter(model_name):
    if model_name == 'SpVit_base':
        cfg = SpVit_base
    else:
        assert("we didn't get model parameter")
    return cfg

def build_model(model_name, model_cfg, train_cfg, device):
    model_name = model_name.split('_')[0]
    if model_name == 'SpVit':
        model = SpVit(image_size=model_cfg.image_size, num_sample=model_cfg.num_sample, patch_size=model_cfg.patch_size, embed_dim=model_cfg.embed_dim, depth=model_cfg.depth,
                      num_head=model_cfg.num_head, num_classes=train_cfg.num_classes, critic_lr=train_cfg.RL_batch_size * train_cfg.critic_lr_factor, actor_lr=train_cfg.RL_batch_size * train_cfg.actor_lr_factor,
                      device=device)
        rpb = ReplayBuffer(train_cfg.ReplayBufferCapacity, train_cfg.RL_batch_size)
    else:
        assert("Didn't find model")
    return model, rpb

def get_dataloader(dataset_name, train_cfg, model_config):
    if dataset_name == 'CIFAR10':
        
        Image_augment = AutoAugment(AutoAugmentPolicy.CIFAR10)
        
        train_transform = Compose([
            Image_augment,
            ToTensor(),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        test_transform = Compose([
            ToTensor(),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        train_dataset = CIFAR10(root='C:\Worksapce\project', train=True, download=False, transform=train_transform)
        test_dataset = CIFAR10(root='C:\Worksapce\project', train=True, download=False, transform=test_transform)
        train_dataloader = DataLoader(train_dataset, train_cfg.vit_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, train_cfg.vit_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        assert("Didn't find model")
    return train_dataloader, test_dataloader

def visualize_in_one_plot(train_trend, test_trend, trend_name):
    x = list(range(len(train_trend)))
    plt.plot(x, train_trend, label='train', color='r')
    plt.plot(x, test_trend, label='test', color='b')
    plt.title(trend_name)
    plt.legend(loc='best')
    plt.savefig(model_name + '_' + dataset_name + '_' + trend_name + '.jpg')
    plt.show()

def save_at_csv(train_accuracy_trend, test_accuracy_trend, train_loss_trend, test_loss_trend):
    trend_pool = []
    trend_pool.append(train_accuracy_trend)
    trend_pool.append(test_accuracy_trend)
    trend_pool.append(train_loss_trend)
    trend_pool.append(test_loss_trend)
    trend_pool = list(map(list, zip(*trend_pool)))
    name = ['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss']
    df = pd.DataFrame(columns=name, data=trend_pool)
    df.to_csv('./' + training_config_name + '.csv')

if __name__ == "__main__": 
    train_cfg = get_train_parameter(training_config_name)
    model_cfg = get_model_parameter(model_name)
    loss_function = CrossEntropyLoss()

    model, rpb = build_model(model_name, model_cfg, train_cfg, device=device)
    train_dataloader, test_dataloader = get_dataloader(dataset_name, train_cfg, model_cfg)
    optimizer= Adam(model.parameters(), lr=train_cfg.vit_batch_size * train_cfg.vit_lr_factor)


    train_accurate_trend, train_loss_trend, val_accurate_trend, val_loss_trend = train(model, train_dataloader, test_dataloader, rpb, loss_function, optimizer, epochs, device)
    visualize_in_one_plot(train_accurate_trend, val_accurate_trend, "accuracy")
    visualize_in_one_plot(train_loss_trend, val_loss_trend, "loss")

    save_at_csv(train_accurate_trend, val_accurate_trend, train_loss_trend, val_loss_trend)



