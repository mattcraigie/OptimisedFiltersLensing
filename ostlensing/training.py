import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt

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

    def test(self, test_loader, load_best=True):
        if load_best:
            self.model.load_state_dict(self.best_model_params)
        # for test, we must also specify a test_loader
        test_loss = self._run_epoch(test_loader, self.val_criterion, mode='eval')

        if self.ddp:
            test_loss = torch.tensor(test_loss).to(self.device)
            dist.reduce(test_loss, dst=0, op=dist.ReduceOp.SUM)
            if self.device != 0:
                test_loss = test_loss.item() / dist.get_world_size()

        return test_loss

    def plot_filters(self, nrows, ncols, save_path):
        filters = self.model.filters.filter_tensor.cpu().detach().numpy()

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        for i in range(nrows):
            for j in range(ncols):
                try:
                    axes[i, j].imshow(filters[i * ncols + j, 0, :, :])
                    axes[i, j].axis('off')
                except IndexError:
                    continue
        plt.savefig(save_path)

    def plot_losses(self, save_path):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.train_losses, label='train')
        ax.plot(self.val_losses, label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.savefig(save_path)

    def save_model(self, save_path):
        torch.save(self.best_model_params, save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def plot_predictions(self, test_loader, save_path, num_samples=None):
        if num_samples is None:
            num_samples = len(self.train_loader.dataset)
        num_targets = self.train_loader.dataset.target.shape[1]

        fig, axes = plt.subplots(num_targets, 2, figsize=(4, num_targets*2))

        train_data = self.train_loader.dataset.data[:num_samples]
        train_target = self.train_loader.dataset.target[:num_samples]
        train_pred = batch_apply(train_data, 32, self.model, self.device).cpu().detach().numpy()

        test_data = test_loader.dataset.data[:num_samples]
        test_target = test_loader.dataset.target[:num_samples]
        test_pred = batch_apply(test_data, 32, self.model, self.device).cpu().detach().numpy()

        for i in range(num_targets):
            axes[i, 0].scatter(train_target[:, i], train_pred[:, i])
            axes[i, 1].scatter(test_target[:, i], test_pred[:, i])
            axes[i, 0].set_xlabel('Target')
            axes[i, 0].set_ylabel('Prediction')
            axes[i, 1].set_xlabel('Target')
            axes[i, 1].set_ylabel('Prediction')
            axes[i, 0].set_aspect('equal')
            axes[i, 1].set_aspect('equal')
            axes[i, 0].plot([0, 1], [0, 1], transform=axes[i, 0].transAxes, c='k')
            axes[i, 1].plot([0, 1], [0, 1], transform=axes[i, 0].transAxes, c='k')

        axes[0, 0].set_title('Train')
        axes[0, 1].set_title('Test')
        plt.tight_layout()

        plt.savefig(save_path)



