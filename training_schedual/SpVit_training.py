import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.profiler import profile, record_function, ProfilerActivity

def train(models, training_dataloader, val_dataloader, replaybuffer, loss_function, optimizer, epochs, device):
    models = models.to(device)
    total_training_image = len(training_dataloader.dataset)
    total_val_image = len(val_dataloader.dataset)
    train_accurate_trend = []
    train_loss_trend = []
    val_accurate_trend = []
    val_loss_trend = []
    for epoch in range(epochs):
        total_accurate = 0
        total_loss = 0
        models.train()
        for image, target in training_dataloader:
            batch_size = image.shape[0]
            image = image.to(device)
            target = target.to(device)
            logit, action = models(image)
            loss = loss_function(logit, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Vit part finish

            _, out_class = torch.max(logit, dim=1)
            reward = (out_class == target).int()
            replaybuffer.add(image, action, reward)
            if len(replaybuffer.buffer) == replaybuffer.buffer.maxlen:
                rl_state, rl_action, rl_reward = replaybuffer.sample()
                models.RL_agent.update(rl_state, rl_action, rl_reward)
            total_accurate += reward.sum().item()
            total_loss += loss.item()
        print(f"Training {epoch + 1} / {epochs}:", "Accuracy: %.4f, loss:%.4f(%.4f)" % (total_accurate / total_training_image, loss, total_loss / len(training_dataloader)))
        train_accurate_trend.append(total_accurate / total_training_image)
        train_loss_trend.append(total_loss / len(training_dataloader))
        models.eval()
        total_accurate = 0
        total_loss = 0
        for image, target in val_dataloader:
            batch_size = image.shape[0]
            image = image.to(device)
            target = target.to(device)
            with torch.no_grad():
                logit, action = models(image)
                loss = loss_function(logit, target)
            _, out_class = torch.max(logit, dim=1)
            reward = (out_class == target).int()
            total_accurate += reward.sum().item()
            total_loss += loss.item()
        print(f"Eval {epoch + 1} / {epochs}:", "Accuracy: %.4f, loss:%.4f(%.4f)" % (total_accurate / total_val_image, loss, total_loss / len(val_dataloader)))
        val_accurate_trend.append(total_accurate / total_val_image)
        val_loss_trend.append(total_loss / len(val_dataloader))
    return train_accurate_trend, train_loss_trend, val_accurate_trend, val_loss_trend


            
