import torch
import torch.nn as nn


# ~~~ Training Functions ~~~ #

def mse_and_admissibility(output, target, model, weighting=1.0):
    loss = nn.functional.mse_loss(output, target)
    loss += weighting * model.filters.filter_tensor[1, 0, 0, 0]**2  # (zero-freq fourier mode / mean config space)
    return loss


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target, model)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader.dataset)


def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target, model)
            val_loss += loss.item()
    return val_loss / len(val_loader.dataset)


def train_loop(model, optimizer, criterion, train_loader, val_loader, device, epochs=10):
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_model_params = None
    best_filters = None

    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        val_loss = validate(model, criterion, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_params = model.state_dict()
            best_filters = model.filters.filter_tensor

    return train_losses, val_losses, best_model_params, best_filters


def batch_apply(data, bs, func, device):
    results = []
    num_batches = data.shape[0] // bs
    num_batches = num_batches if data.shape[0] % bs == 0 else num_batches + 1
    for i in range(num_batches):
        x = data[bs*i:bs*(i+1)]
        x = x.to(device)
        results.append(func(x))
    return torch.cat(results, dim=0)
