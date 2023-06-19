import matplotlib.pyplot as plt
import torch
import os


class Plotter:
    def __init__(self):
        self.model = None
        self.losses = None
        self.predictions = None
        self.targets = None

    def load_folder(self, folder):
        self.model = torch.load(os.path.join(folder, 'model.pt'))
        self.losses = torch.load(os.path.join(folder, 'losses.pt'))
        self.predictions = torch.load(os.path.join(folder, 'predictions.pt'))
        self.targets = torch.load(os.path.join(folder, 'targets.pt'))

    def plot_filters(self, nrows, ncols, save_path):
        if self.model is None:
            raise ValueError('Model not set. Call load_folder first.')
        try:
            filters = self.model.filters.filter_tensor.cpu().detach().numpy()
        except AttributeError:
            raise AttributeError('Model does not have filters.')

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        for i in range(nrows):
            for j in range(ncols):
                try:
                    axes[i, j].imshow(filters[i * ncols + j, 0, :, :])
                    axes[i, j].axis('off')
                except IndexError:
                    continue
        plt.savefig(save_path)

    def plot_losses(self, save_path=None):
        if self.losses is None:
            raise ValueError('Losses not set. Call load_folder first.')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.losses['train'], label='train')
        ax.plot(self.losses['val'], label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_predictions(self, save_path=None, show_val=True, num_samples=None):
        if self.predictions is None or self.targets is None:
            raise ValueError('Predictions or targets not set. Call load_folder first.')
        num_targets = self.targets.shape[1]

        fig, axes = plt.subplots(num_targets, 2, figsize=(4, num_targets*2))

        for i in range(num_targets):
            # train (and val)
            axes[i, 0].scatter(self.targets['train'][:num_samples, i],
                               self.predictions['train'][:num_samples, i], c='blue')
            if show_val:
                axes[i, 0].scatter(self.targets['val'][:num_samples, i],
                                   self.predictions['val'][:num_samples, i], c='green')
            axes[i, 0].set_xlabel('Target')
            axes[i, 0].set_ylabel('Prediction')
            axes[i, 0].set_aspect('equal')
            axes[i, 0].plot([0, 1], [0, 1], transform=axes[i, 0].transAxes, c='black')

            # test
            axes[i, 1].scatter(self.targets['test'][:num_samples, i],
                               self.predictions['test'][:num_samples, i], c='pink')
            axes[i, 1].set_xlabel('Target')
            axes[i, 1].set_ylabel('Prediction')
            axes[i, 1].set_aspect('equal')
            axes[i, 1].plot([0, 1], [0, 1], transform=axes[i, 0].transAxes, c='black')

        axes[0, 0].set_title('Train')
        axes[0, 1].set_title('Test')
        plt.tight_layout()

        plt.savefig(save_path)


