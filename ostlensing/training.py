import torch
import torch.nn as nn
import torch.distributed as dist


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


# ~~~ Trainer Class ~~~ #

class Trainer:
    # implement gradient averaging
    def __init__(self, model, optimizer, train_criterion, val_criterion, train_loader, val_loader, device,
                 ddp=False):
        self.model = model
        self.optimizer = optimizer
        self.train_criterion = train_criterion
        self.val_criterion = val_criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
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

        # accumulate and average across GPUs if using DDP
        if self.ddp:
            avg_loss = dist.reduce(avg_loss, dst=0, op=dist.ReduceOp.SUM) / dist.get_world_size()

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

            if self.ddp and self.device != 0:
                continue

            # only save losses for rank 0
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_params = self.model.state_dict()

    def get_best_model(self):
        self.model.load_state_dict(self.best_model_params)
        return self.model

    def test(self, test_loader, load_best=True):
        if load_best:
            self.model.load_state_dict(self.best_model_params)
        # for test, we must also specify a test_loader
        test_loss = self._run_epoch(test_loader, self.val_criterion, mode='eval')
        return test_loss




