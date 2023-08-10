import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import numpy as np


def plot_scaling(scaling_paths, save_path=None, logy=True, logx=True, labels=None, colours=None, transform_std=None,
                 show_repeats=False, quantiles=True):

    if labels is None:
        labels = [str(i) for i in range(len(scaling_paths))]

    if colours is None:
        colours = ['C' + str(i) for i in range(len(scaling_paths))]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    for i in range(len(scaling_paths)):
        scaling_df = pd.read_csv(scaling_paths[i])
        mse_norm = scaling_df.iloc[:, 1:]
        rmse_norm = np.sqrt(mse_norm)

        # rescale all values to data units if transform is provided
        rmse = rmse_norm * transform_std if transform_std is not None else rmse_norm

        if quantiles:
            rmse_mid = np.median(rmse, axis=1)
            rmse_low = np.quantile(rmse, 0.25, axis=1)
            rmse_high = np.quantile(rmse, 0.75, axis=1)
        else:
            rmse_mid = np.mean(rmse, axis=1)
            rmse_low = rmse_mid - np.std(rmse, axis=1)
            rmse_high = rmse_mid + np.std(rmse, axis=1)


        x = scaling_df['data_subset']
        ax.plot(x, rmse_mid, linewidth=4, label=labels[i], c=colours[i])
        # scatter with a square marker
        ax.scatter(x, rmse_mid, c=colours[i], s=35, marker='s')
        ax.plot(x, rmse_low, alpha=0.3, linewidth=1.5, c=colours[i])
        ax.plot(x, rmse_high, alpha=0.3, linewidth=1.5, c=colours[i])
        ax.fill_between(x, rmse_low, rmse_high, alpha=0.15, color=colours[i])

        if show_repeats:
            for j in range(rmse.shape[0]):
                ax.scatter([x[j] for _ in range(rmse.shape[1])], rmse.iloc[j], c=colours[i], alpha=0.4, marker='x')

    ax.set_xlabel('Number of Training Cosmologies', fontsize=20)
    ax.set_ylabel('Test RMSE', fontsize=20)

    # create a legend with the labels fontsize 16
    ax.legend(fontsize=16)




    if logy and not logx:
        plt.semilogy()
    if logx and not logy:
        plt.semilogx()
    if logy and logx:
        plt.loglog()
    if not logy:
        plt.ylim(bottom=0)

    # set tick label sizes after logging
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)


    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_improvement(baseline_path, other_paths, save_path=None, logx=True, labels=None, colours=None,
                     show_repeats=False, quantiles=True):
    if labels is None:
        labels = [str(i) for i in range(len(other_paths))]

    if colours is None:
        colours = ['C' + str(i) for i in range(len(other_paths))]

    # baseline

    baseline_df = pd.read_csv(baseline_path)
    baseline_x = baseline_df['data_subset']
    baseline_mse = baseline_df.iloc[:, 1:]
    baseline_rmse = np.sqrt(baseline_mse)
    if quantiles:
        baseline_rmse = np.median(baseline_rmse, axis=1)
    else:
        baseline_rmse = np.mean(baseline_rmse, axis=1)

    print(baseline_rmse)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    for i in range(len(other_paths)):
        scaling_df = pd.read_csv(other_paths[i])
        mse = scaling_df.iloc[:, 1:]
        rmse = np.sqrt(mse)

        # calculate the fractional improvement from the baseline
        x = scaling_df['data_subset']
        includes = indices = np.where(np.in1d(x.values, baseline_x.values))[0]
        print(includes)

        a = rmse.values
        b = baseline_rmse.values[includes, None]
        improvement = (a - b) / b * 100

        if quantiles:
            improvement_mid = np.median(improvement, axis=1)
            improvement_low = np.quantile(improvement, 0.25, axis=1)
            improvement_high = np.quantile(improvement, 0.75, axis=1)
        else:
            improvement_mid = np.mean(improvement, axis=1)
            improvement_low = improvement_mid - np.std(improvement, axis=1)
            improvement_high = improvement_mid + np.std(improvement, axis=1)

        ax.plot(x, improvement_mid, linewidth=4, label=labels[i], c=colours[i])
        # scatter with a square marker
        ax.scatter(x, improvement_mid, c=colours[i], s=35, marker='s')
        ax.plot(x, improvement_low, alpha=0.3, linewidth=1.5, c=colours[i])
        ax.plot(x, improvement_high, alpha=0.3, linewidth=1.5, c=colours[i])
        ax.fill_between(x, improvement_low, improvement_high, alpha=0.15, color=colours[i])

        if show_repeats:
            for j in range(improvement.shape[0]):
                ax.scatter([x[j] for _ in range(improvement.shape[1])], improvement[j], c=colours[i], alpha=0.4,
                           marker='x')

    ax.set_xlabel('Number of Training Cosmologies', fontsize=20)
    ax.set_ylabel('Change over Baseline (%)', fontsize=20)

    # create a legend with the labels fontsize 16
    ax.legend(fontsize=16)

    if logx:
        plt.semilogx()

    # set tick label sizes after logging
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

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

    def plot_predictions(self, flat_plot=False, save_path=None, param_names=None, param_transforms=None, manual_ylims=None):
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


        figsize = (8, num_targets*4)

        fig, axes = plt.subplots(num_targets, 2, figsize=figsize)

        if num_targets == 1:
            axes = axes.reshape(1, 2)  # Reshape the axes to be 2D to handle the 1 parameter case

        for i in range(num_targets):
            # train (and val)
            x_train = transform(self.targets['train'][:, i], i)
            y_train = transform(self.predictions['train'][:, i], i)

            x_val = transform(self.targets['val'][:, i], i)
            y_val = transform(self.predictions['val'][:, i], i)

            x_test = transform(self.targets['test'][:, i], i)
            y_test = transform(self.predictions['test'][:, i], i)

            min_val, max_val = min(x_test), max(x_test)

            if not flat_plot:
                axes[i, 0].scatter(x_train, y_train, c='cornflowerblue', alpha=0.5, label='train')
                axes[i, 0].scatter(x_val, y_val, c='green', marker='x', alpha=0.5, label='validation')
                axes[i, 1].scatter(x_test, y_test, c='deeppink', alpha=0.5)

                axes[i, 0].plot([0, 1], [0, 1], transform=axes[i, 0].transAxes, c='black')
                axes[i, 1].plot([0, 1], [0, 1], transform=axes[i, 1].transAxes, c='black')

                for p in [0, 1]:
                    axes[i, p].set_xlim(min_val, max_val)
                    axes[i, p].set_ylim(min_val, max_val)
                    axes[i, p].set_ylabel('Prediction {}'.format(param_names[i]))

            else:
                axes[i, 0].scatter(x_train, y_train - x_train, c='cornflowerblue', alpha=0.5, label='train')
                axes[i, 0].scatter(x_val, y_val - x_val, c='green', marker='x', alpha=0.5, label='validation')
                axes[i, 1].scatter(x_test, y_test - x_test, c='deeppink', alpha=0.5)

                axes[i, 0].plot([0, 1], [0.5, 0.5], transform=axes[i, 0].transAxes, c='black')
                axes[i, 1].plot([0, 1], [0.5, 0.5], transform=axes[i, 1].transAxes, c='black')

                ylims = np.abs(np.max(x_test) - np.min(x_test)) * 0.3 if manual_ylims is None else manual_ylims[i]
                for p in [0, 1]:

                    axes[i, p].set_xlim(min_val, max_val)
                    axes[i, p].set_ylim(-ylims, ylims)
                    axes[i, p].set_ylabel('Prediction {} - Target {}'.format(param_names[i], param_names[i]))


            axes[i, 0].legend()
            axes[i, 0].set_xlabel('Target {}'.format(param_names[i]))
            axes[i, 1].set_xlabel('Target {}'.format(param_names[i]))

            # text showing test RMSE on top of each test plot
            test_loss = np.sqrt(np.mean((y_test - x_test)**2))
            axes[i, 1].text(0.05, 0.95, 'RMSE: {:.3e}'.format(test_loss), transform=axes[i, 1].transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        axes[0, 0].set_title('Train')
        axes[0, 1].set_title('Test')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()