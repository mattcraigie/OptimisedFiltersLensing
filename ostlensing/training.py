import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from scattering_transform.filters import scale2size
import os
import copy

# ~~~ Loss Functions ~~~ #


def mse_and_admissibility(output, target, model, weighting=1.0, edge_constraints=False):
    loss = nn.functional.mse_loss(output, target)
    ft = model.filters.filter_tensor

    loss += weighting * ft[:, 0, 0, 0].abs().sum()  # (zero-freq fourier mode / mean config space)

    if edge_constraints:

        size = ft.shape[-1]

        for i in range(ft.shape[0]):
            # also add a loss term to penalise the model for having nonzero edges
            scaled_size = scale2size(size, i)
            k = ft[i, 0]
            half = size // 2
            half_scaled = scaled_size // 2
            k = torch.fft.fftshift(k)
            start = half - half_scaled
            end = half + half_scaled
            edge_loss = k[start, start:end].abs().sum() + \
                        k[start:end, start].abs().sum() + \
                        k[start + 1, start + 1:end].abs().sum() + \
                        k[start + 1:end, start + 1].abs().sum() + \
                        k[end - 1, start + 1:end].abs().sum() + \
                        k[start + 1:end, end - 1].abs().sum()

            loss += weighting * edge_loss

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
    targets = []
    for x, t in dataloader:
        x = x.to(device)
        t = t.to(device)
        results.append(func(x))
        targets.append(t)
    return torch.cat(results, dim=0), torch.cat(targets, dim=0)


# ~~~ Trainer Class ~~~ #

class Trainer:
    # implement gradient averaging
    def __init__(self, regressor, optimizer, train_criterion, val_criterion, train_loader, val_loader, test_loader, device,
                 ddp=False):

        self.regressor = regressor
        self.optimizer = optimizer
        self.train_criterion = train_criterion
        self.val_criterion = val_criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.best_loss = float('inf')
        self.best_regressor_params = None
        self.train_losses = []
        self.val_losses = []
        self.ddp = ddp  # distributed data parallel

        self.train_pred = None
        self.val_pred = None
        self.test_pred = None

        self.train_targets = None
        self.val_targets = None
        self.test_targets = None

        self.num_train_samples = len(self.train_loader.dataset)
        self.num_val_samples = len(self.val_loader.dataset)
        self.num_test_samples = len(self.test_loader.dataset)

        self.best_test_loss = None

    def _run_epoch(self, loader, criterion, mode='train'):
        if mode == 'train':
            self.regressor.train()
        else:
            self.regressor.eval()

        total_loss = 0

        with torch.set_grad_enabled(mode == 'train'):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.regressor(data)
                loss = criterion(output, target, self.regressor)

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        return total_loss

    def train(self):
        return self._run_epoch(self.train_loader, self.train_criterion, mode='train')

    def validate(self):
        return self._run_epoch(self.val_loader, self.val_criterion, mode='eval')

    def train_loop(self, epochs=10):
        for epoch in range(0, epochs):

            self.train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)

            sum_train_loss = self.train()
            sum_val_loss = self.validate()

            # Everything beyond here is just for loss visualisation and validation purposes

            if self.ddp:
                sum_train_loss = torch.tensor(sum_train_loss).to(self.device)
                dist.reduce(sum_train_loss, dst=0, op=dist.ReduceOp.SUM)  # this operation is inplace and returns to rank 0

                val_loss = torch.tensor(sum_val_loss).to(self.device)
                dist.reduce(val_loss, dst=0, op=dist.ReduceOp.SUM)

                train_loss = sum_train_loss.item() / self.num_train_samples
                val_loss = val_loss.item() / self.num_val_samples

            else:
                train_loss = sum_train_loss / self.num_train_samples
                val_loss = sum_val_loss / self.num_val_samples

            if self.device == 0 or not self.ddp:
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_regressor_params = copy.deepcopy(self.regressor.state_dict())

    def load_best_model(self):
        self.regressor.load_state_dict(self.best_regressor_params)

    def make_predictions(self):
        self.regressor.eval()
        with torch.no_grad():
            self.train_pred, self.train_targets = dataloader_apply(self.train_loader, self.regressor, self.device)
            self.val_pred, self.val_targets = dataloader_apply(self.val_loader, self.regressor, self.device)
            self.test_pred, self.test_targets = dataloader_apply(self.test_loader, self.regressor, self.device)

            if self.ddp:
                def gatherer(x):
                    torch.cuda.set_device(self.device)
                    gathered_x = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_x, x)
                    gathered_x = torch.cat(gathered_x, dim=0)
                    # gathered_x = torch.unique(gathered_x, dim=0, sorted=False)
                    return gathered_x

                self.train_pred = gatherer(self.train_pred)
                self.val_pred = gatherer(self.val_pred)
                self.test_pred = gatherer(self.test_pred)
                self.train_targets = gatherer(self.train_targets)
                self.val_targets = gatherer(self.val_targets)
                self.test_targets = gatherer(self.test_targets)

            self.train_pred = self.train_pred.cpu()
            self.val_pred = self.val_pred.cpu()
            self.test_pred = self.test_pred.cpu()
            self.train_targets = self.train_targets.cpu()
            self.val_targets = self.val_targets.cpu()
            self.test_targets = self.test_targets.cpu()

    def save_model(self, save_path):
        regressor = self.regressor.module if self.ddp else self.regressor
        try:
            regressor.save(save_path)
        except AttributeError:
            pass  # model does not have a save method, no worries

    def save_losses(self, save_path):
        torch.save({'train': self.train_losses, 'val': self.val_losses}, save_path)

    def save_predictions(self, save_path):
        torch.save({'train': self.train_pred, 'val': self.val_pred, 'test': self.test_pred}, save_path)

    def save_targets(self, save_path):
        torch.save({'train': self.train_targets, 'val': self.val_targets, 'test': self.test_targets}, save_path)

    def load_model(self, load_path):
        self.regressor.load_state_dict(torch.load(load_path))

    def save_filters(self, save_path):
        regressor = self.regressor.module if self.ddp else self.regressor
        try:
            filters = regressor.model.filters.filter_tensor.cpu().detach()
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





