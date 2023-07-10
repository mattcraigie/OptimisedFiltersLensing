import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import numpy as np


def plot_scaling(scaling_paths, save_path=None, logy=True, logx=True, labels=None, colours=None, transform_sigma=None,
                 show_repeats=False):

    if labels is None:
        labels = [str(i) for i in range(len(scaling_paths))]

    if colours is None:
        colours = ['C' + str(i) for i in range(len(scaling_paths))]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(scaling_paths)):
        scaling_df = pd.read_csv(scaling_paths[i])
        data = scaling_df.iloc[:, 1:]

        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        upper = mean + std
        lower = mean - std
        root_mean, root_lower, root_upper = np.sqrt(mean), np.sqrt(lower), np.sqrt(upper)

        # rescale all values to data units if transform is provided
        if transform_sigma is not None:
            root_mean, root_lower, root_upper = root_mean * transform_sigma, root_lower * transform_sigma, root_upper * transform_sigma

        x = scaling_df['data_subset']
        ax.plot(x, root_mean, linewidth=4, label=labels[i], c=colours[i])
        ax.scatter(x, root_mean, c=colours[i])
        ax.plot(x, root_lower, alpha=0.4, linewidth=1, c=colours[i])
        ax.plot(x, root_upper, alpha=0.4, linewidth=1, c=colours[i])
        ax.fill_between(x, root_lower, root_upper, alpha=0.2, color=colours[i])

        if show_repeats:
            for j in range(data.shape[0]):
                ax.scatter([x[j] for _ in range(data.shape[1])], np.sqrt(data.iloc[j]), c=colours[i], alpha=0.4, marker='x')

    ax.set_xlabel('Number of Training Cosmologies', fontsize=16)
    ax.set_ylabel('Test Sample RMSE ($\\approx 1\\sigma$ constraint)', fontsize=16)
    plt.legend()

    if logy and not logx:
        plt.semilogy()
    if logx and not logy:
        plt.semilogx()
    if logy and logx:
        plt.loglog()
    if not logy:
        plt.ylim(bottom=0)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


class ModelPlotter:
    def __init__(self):
        self.model = None
        self.losses = None
        self.predictions = None
        self.targets = None
        self.filters = None

    def load_folder(self, folder):
        try:
            self.model = torch.load(os.path.join(folder, 'model.pt'))
        except FileNotFoundError:
            pass  # it's precalc - no model
        self.losses = torch.load(os.path.join(folder, 'losses.pt'))
        self.predictions = torch.load(os.path.join(folder, 'predictions.pt'))
        self.targets = torch.load(os.path.join(folder, 'targets.pt'))

        try:
            self.filters = torch.load(os.path.join(folder, 'filters.pt'))
        except FileNotFoundError:
            pass

    def plot_filters(self, save_path=None):

        if self.filters is None:
            raise AttributeError('Model does not have filters.')

        filters = self.filters
        filter_size = filters.shape[-1]

        ncols = 3
        nrows = filters.shape[0]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
        for j in range(filters.shape[0]):
            k = filters[j, 0, :, :]
            k_full = k.clone()

            # keep the corners of the filters
            k = torch.fft.fftshift(k)
            keep_size = filter_size // 2**j
            half = keep_size // 2
            k = k[64 - half:64 + half, 64 - half:64 + half]

            axes[j, 0].imshow(k)
            axes[j, 0].axis('off')

            x = torch.fft.fft2(k_full)
            x = torch.fft.fftshift(x)
            # keep the middle of the filters
            keep_size = 2**(j+3)
            half = keep_size // 2
            x = x[64 - half:64 + half, 64 - half:64 + half]

            axes[j, 1].imshow(x.real)
            axes[j, 1].axis('off')

            axes[j, 2].imshow(x.imag)
            axes[j, 2].axis('off')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_losses(self, save_path=None, semilogy=True):
        if self.losses is None:
            raise ValueError('Losses not set. Call load_folder first.')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.losses['train'], label='train', linewidth=4)
        ax.plot(self.losses['val'], label='val', linewidth=4)
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Loss', fontsize=16)
        ax.legend()
        if semilogy:
            plt.semilogy()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_predictions(self, flat_plot=False, save_path=None, show_val=True, num_samples=None, param_names=None, param_transforms=None):
        if self.predictions is None or self.targets is None:
            raise ValueError('Predictions or targets not set. Call load_folder first.')

        num_targets = self.targets['train'].shape[1]

        if param_names is None:
            param_names = [str(i) for i in range(num_targets)]

        def transform(x, i):
            if type(x) == torch.Tensor:
                x = x.numpy()
            if param_transforms is None:
                return x
            return (x * param_transforms[i][1]) + param_transforms[i][0]

        if not flat_plot:
            figsize = (8, num_targets*8)
        else:
            figsize = (8, num_targets*4)

        fig, axes = plt.subplots(num_targets, 2, figsize=figsize)

        if num_targets == 1:
            axes = axes.reshape(1, 2)  # Reshape the axes to be 2D to handle the 1 parameter case

        for i in range(num_targets):
            # train (and val)
            x_train = transform(self.targets['train'][:num_samples, i], i)
            y_train = transform(self.predictions['train'][:num_samples, i], i)

            if not flat_plot:
                axes[i, 0].scatter(x_train, y_train, c='cornflowerblue', alpha=0.5, label='train')
            else:
                axes[i, 0].scatter(x_train, y_train - x_train, c='cornflowerblue', alpha=0.5, label='train')

            if show_val:
                x_val = transform(self.targets['val'][:num_samples, i], i)
                y_val = transform(self.predictions['val'][:num_samples, i], i)

                if not flat_plot:
                    axes[i, 0].scatter(x_val, y_val, c='green', marker='x', alpha=0.5, label='validation')
                else:
                    axes[i, 0].scatter(x_val, y_val - x_val, c='green', marker='x', alpha=0.5, label='validation')

            axes[i, 0].set_xlabel('Target {}'.format(param_names[i]))

            if not flat_plot:
                axes[i, 0].plot([0, 1], [0, 1], transform=axes[i, 0].transAxes, c='black')
                axes[i, 0].set_aspect('equal')
                axes[i, 0].set_ylabel('Prediction {}'.format(param_names[i]))
            else:
                axes[i, 0].plot([0, 1], [0.5, 0.5], transform=axes[i, 0].transAxes, c='black')
                axes[i, 0].set_ylabel('Prediction {} - Target {}'.format(param_names[i], param_names[i]))
                ylims = np.abs(np.max(x_train) - np.min(x_train)) * 0.5
                axes[i, 0].set_ylim(-ylims, ylims)

            axes[i, 0].legend()

            # test
            x_test = transform(self.targets['test'][:num_samples, i], i)
            y_test = transform(self.predictions['test'][:num_samples, i], i)

            if not flat_plot:
                axes[i, 1].scatter(x_test, y_test, c='deeppink', alpha=0.5)
                axes[i, 1].plot([0, 1], [0, 1], transform=axes[i, 1].transAxes, c='black')
                axes[i, 1].set_aspect('equal')
                axes[i, 1].set_ylabel('Prediction {}'.format(param_names[i]))
            else:
                axes[i, 1].scatter(x_test, y_test - x_test, c='deeppink', alpha=0.5)
                axes[i, 1].plot([0, 1], [0.5, 0.5], transform=axes[i, 1].transAxes, c='black')
                axes[i, 1].set_ylabel('Prediction {} - Target {}'.format(param_names[i], param_names[i]))
                ylims = np.abs(np.max(x_train) - np.min(x_train)) * 0.5
                axes[i, 1].set_ylim(-ylims, ylims)

            axes[i, 1].set_xlabel('Target {}'.format(param_names[i]))
            # add text showing the rmse in the top left corner of the plot
            test_loss = np.sqrt(np.mean((y_test - x_test)**2))

            # use scientific notation
            axes[i, 1].text(0.05, 0.95, 'RMSE: {:.3e}'.format(test_loss), transform=axes[i, 1].transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        axes[0, 0].set_title('Train')
        axes[0, 1].set_title('Test')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()