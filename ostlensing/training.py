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

    def _run_epoch(self, loader, criterion, mode='train'):
        if mode == 'train':
            self.regressor.train()
        else:
            self.regressor.eval()

        total_loss = 0
        num_samples = len(loader.dataset)

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

        return total_loss, num_samples

    def train(self):
        train_loss = self._run_epoch(self.train_loader, self.train_criterion, mode='train')
        return train_loss

    def validate(self):
        val_loss = self._run_epoch(self.val_loader, self.val_criterion, mode='eval')
        return val_loss

    def train_loop(self, epochs=10):
        for epoch in range(1, epochs + 1):

            sum_train_loss, num_train_samples = self.train()
            sum_val_loss, num_val_samples = self.validate()

            if self.ddp:
                train_loss = torch.tensor(sum_train_loss).to(self.device)
                dist.reduce(train_loss, dst=0, op=dist.ReduceOp.SUM)  # this operation is inplace and returns to rank 0

                val_loss = torch.tensor(sum_val_loss).to(self.device)
                dist.reduce(val_loss, dst=0, op=dist.ReduceOp.SUM)

                num_train_samples = torch.tensor(num_train_samples).to(self.device)
                dist.reduce(num_train_samples, dst=0, op=dist.ReduceOp.SUM)

                num_val_samples = torch.tensor(num_val_samples).to(self.device)
                dist.reduce(num_val_samples, dst=0, op=dist.ReduceOp.SUM)

                if self.device == 0:
                    train_loss = train_loss.item() / num_train_samples.item()
                    val_loss = val_loss.item() / num_val_samples.item()
            else:
                train_loss = sum_train_loss / num_train_samples
                val_loss = sum_val_loss / num_val_samples

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_regressor_params = self.regressor.state_dict()

    def get_best_model(self):
        self.regressor.load_state_dict(self.best_regressor_params)
        return self.regressor

    def test(self, load_best=True):
        if load_best:
            self.regressor.load_state_dict(self.best_regressor_params)
        # for test, we must also specify a test_loader
        # we do not return the average
        sum_test_loss, num_test_samples = self._run_epoch(self.test_loader, self.val_criterion, mode='eval')

        if self.ddp:
            test_loss = torch.tensor(sum_test_loss).to(self.device)
            dist.reduce(test_loss, dst=0, op=dist.ReduceOp.SUM)

            num_test_samples = torch.tensor(num_test_samples).to(self.device)
            dist.reduce(num_test_samples, dst=0, op=dist.ReduceOp.SUM)

            if self.device != 0:
                test_loss = test_loss.item() / num_test_samples.item()
        else:
            test_loss = sum_test_loss / num_test_samples

        return test_loss

    def save_model(self, save_path):
        regressor = self.regressor.module if self.ddp else self.regressor
        try:
            regressor.save(save_path)
        except AttributeError:
            pass  # model does not have a save method, no worries

    def save_losses(self, save_path):
        torch.save({'train': self.train_losses, 'val': self.val_losses}, save_path)

    def save_predictions(self, save_path):
        self.regressor.eval()
        with torch.no_grad():
            train_pred = dataloader_apply(self.train_loader, self.regressor, self.device).cpu().detach().numpy()
            val_pred = dataloader_apply(self.val_loader, self.regressor, self.device).cpu().detach().numpy()
            test_pred = dataloader_apply(self.test_loader, self.regressor, self.device).cpu().detach().numpy()
            torch.save({'train': train_pred, 'val': val_pred, 'test': test_pred}, save_path)

    def save_targets(self, save_path):
        train_targets = self.train_loader.dataset.targets
        val_targets = self.val_loader.dataset.targets
        test_targets = self.test_loader.dataset.targets
        torch.save({'train': train_targets, 'val': val_targets, 'test': test_targets}, save_path)

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





