import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import numpy as np


def plot_scaling(scaling_paths, save_path=None, logy=True, logx=True, labels=None, colours=None, transform_std=None,
                 show_repeats=False, quantiles=True, figsize=(12, 8), ncols=1, param_labels=None):

    if labels is None:
        labels = [str(i) for i in range(len(scaling_paths))]

    if colours is None:
        colours = ['C' + str(i) for i in range(len(scaling_paths))]

    # test run to get the num_params setup for plotting
    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=100)

    for i, scaling_dir in enumerate(scaling_paths):

        repeat_dirs = os.listdir(scaling_dir)
        repeat_dirs = np.sort(repeat_dirs)
        repeat_rmses = []

        for repeat_dir in repeat_dirs:

            subset_dirs = os.listdir(os.path.join(scaling_dir, repeat_dir))
            subset_dirs = np.sort(subset_dirs)
            subset_sizes = []
            subset_rmses = []

            for subset_dir in subset_dirs:

                subset_sizes.append(int(subset_dir[7:]))

                targets = torch.load(os.path.join(scaling_dir, repeat_dir, subset_dir, 'targets.pt'))
                predictions = torch.load(os.path.join(scaling_dir, repeat_dir, subset_dir, 'predictions.pt'))
                num_params = targets['test'].shape[1]

                param_rmses = []
                for j in range(num_params):
                    targs_test_j = targets['test'][:, j].numpy()
                    preds_test_j = predictions['test'][:, j].numpy()
                    rmse_j = np.sqrt(np.mean((preds_test_j - targs_test_j)**2))
                    param_rmses.append(rmse_j)

                subset_rmses.append(np.array(param_rmses))
            print(subset_rmses)
            repeat_rmses.append(subset_rmses)

        subset_sizes = np.array(subset_sizes)  # shape (subsets,)
        all_rmse = np.stack(repeat_rmses)  # shape (repeats, subsets, params)


        if transform_std is not None:
            transform_std = np.array(transform_std)[None, None, :]
            all_rmse *= transform_std

        if quantiles:
            rmse_mid = np.median(all_rmse, axis=0)
            rmse_low = np.quantile(all_rmse, 0.25, axis=0)
            rmse_high = np.quantile(all_rmse, 0.75, axis=0)
        else:
            rmse_mid = np.mean(all_rmse, axis=0)
            rmse_low = rmse_mid - np.std(all_rmse, axis=0)
            rmse_high = rmse_mid + np.std(all_rmse, axis=0)

        # rmses are shape (subsets, params)

        for j in range(num_params):
            axes[j].plot(subset_sizes, rmse_mid[:, j], linewidth=4, label=labels[i], c=colours[i])
            # scatter with a square marker
            axes[j].scatter(subset_sizes, rmse_mid[:, j], c=colours[i], s=35, marker='s')
            axes[j].plot(subset_sizes, rmse_low[:, j], alpha=0.3, linewidth=1.5, c=colours[i])
            axes[j].plot(subset_sizes, rmse_high[:, j], alpha=0.3, linewidth=1.5, c=colours[i])
            axes[j].fill_between(subset_sizes, rmse_low[:, j], rmse_high[:, j], alpha=0.15, color=colours[i])

            if show_repeats:
                for k in range(all_rmse.shape[1]):
                    axes[j].scatter([subset_sizes[k] for _ in range(all_rmse.shape[0])], all_rmse[:, k, j], c=colours[i], alpha=0.4, marker='x')

            axes[j].set_xlabel('Number of Training Cosmologies', fontsize=16)

            if param_labels is not None:
                axes[j].set_title(param_labels[j])

            if logx and logy:
                print('should be loglog')
                axes[j].loglog()
            elif logx:
                axes[j].semilogx()
            elif logy:
                axes[j].semilogy()

            # Set tick label sizes after logging
            axes[j].tick_params(axis='both', which='major', labelsize=12)
            axes[j].tick_params(axis='both', which='minor', labelsize=12)



        axes[0].legend(fontsize=12)
        axes[0].set_ylabel('Test RMSE', fontsize=16)




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