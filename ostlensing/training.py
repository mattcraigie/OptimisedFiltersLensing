import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
import os

# ~~~ Loss Functions ~~~ #

def mse_and_admissibility(output, target, model, weighting=1.0):
    loss = nn.functional.mse_loss(output, target)
    loss += weighting * model.filters.filter_tensor[1, 0, 0, 0]**2  # (zero-freq fourier mode / mean config space)
    return loss


def mse(output, target, model):
    return nn.functional.mse_loss(output, target)


def batch_apply(data, bs, func, device):
    results = []
    num_batches = data.shape[0] // bs
    num_batches = num_batches if data.shape[0] % bs == 0 else num_batches + 1
    for i in range(num_batches):
        x = data[bs*i:bs*(i+1)]
        x = x.to(device)
        results.append(func(x))
    return torch.cat(results, dim=0)


def dataloader_apply(dataloader, func, device):
    results = []
    for x, _ in dataloader:
        x = x.to(device)
        results.append(func(x))
    return torch.cat(results, dim=0)


# ~~~ Trainer Class ~~~ #

class Trainer:
    # implement gradient averaging
    def __init__(self, model, optimizer, train_criterion, val_criterion, train_loader, val_loader, test_loader, device,
                 ddp=False):
        self.model = model
        self.optimizer = optimizer
        self.train_criterion = train_criterion
        self.val_criterion = val_criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.best_loss = float('inf')
        self.best_model_params = None
        self.train_losses = []
        self.val_losses = []
        self.ddp = ddp  # distributed data parallel

    def _run_epoch(self, loader, criterion, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        num_samples = len(loader.dataset)

        with torch.set_grad_enabled(mode == 'train'):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target, self.model)

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / num_samples

        return avg_loss

    def train(self):
        train_loss = self._run_epoch(self.train_loader, self.train_criterion, mode='train')
        return train_loss

    def validate(self):
        val_loss = self._run_epoch(self.val_loader, self.val_criterion, mode='eval')
        return val_loss

    def train_loop(self, epochs=10):
        for epoch in range(1, epochs + 1):

            train_loss = self.train()
            val_loss = self.validate()

            if self.ddp:
                train_loss = torch.tensor(train_loss).to(self.device)
                val_loss = torch.tensor(val_loss).to(self.device)

                dist.reduce(train_loss, dst=0, op=dist.ReduceOp.SUM)  # this operation is inplace and returns to rank 0
                dist.reduce(val_loss, dst=0, op=dist.ReduceOp.SUM)

                if self.device == 0:
                    train_loss = train_loss.item() / dist.get_world_size()
                    val_loss = val_loss.item() / dist.get_world_size()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_params = self.model.state_dict()

    def get_best_model(self):
        self.model.load_state_dict(self.best_model_params)
        return self.model

    def test(self, load_best=True):
        if load_best:
            self.model.load_state_dict(self.best_model_params)
        # for test, we must also specify a test_loader
        test_loss = self._run_epoch(self.test_loader, self.val_criterion, mode='eval')

        if self.ddp:
            test_loss = torch.tensor(test_loss).to(self.device)
            dist.reduce(test_loss, dst=0, op=dist.ReduceOp.SUM)
            if self.device != 0:
                test_loss = test_loss.item() / dist.get_world_size()

        return test_loss

    def save_model(self, save_path):
        model = self.model.module if self.ddp else self.model
        model.save(save_path)

    def save_losses(self, save_path):
        torch.save({'train': self.train_losses, 'val': self.val_losses}, save_path)

    def save_predictions(self, save_path):
        self.model.eval()
        with torch.no_grad():
            train_pred = dataloader_apply(self.train_loader, self.model, self.device).cpu().detach().numpy()
            val_pred = dataloader_apply(self.val_loader, self.model, self.device).cpu().detach().numpy()
            test_pred = dataloader_apply(self.test_loader, self.model, self.device).cpu().detach().numpy()
            torch.save({'train': train_pred, 'val': val_pred, 'test': test_pred}, save_path)

    def save_targets(self, save_path):
        train_targets = self.train_loader.dataset.targets
        val_targets = self.val_loader.dataset.targets
        test_targets = self.test_loader.dataset.targets
        torch.save({'train': train_targets, 'val': val_targets, 'test': test_targets}, save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def save_filters(self, save_path):
        model = self.model.module if self.ddp else self.model
        try:
            filters = model.filters.filter_tensor.cpu().detach()
            torch.save(filters, save_path)
        except AttributeError:
            # model doesn't have filters, so don't save anything
            pass

    def save_all(self, save_path):
        self.save_model(os.path.join(save_path, 'model.pt'))
        self.save_losses(os.path.join(save_path, 'losses.pt'))
        self.save_filters(os.path.join(save_path, 'filters.pt'))
        self.save_predictions(os.path.join(save_path, 'predictions.pt'))
        self.save_targets(os.path.join(save_path, 'targets.pt'))





