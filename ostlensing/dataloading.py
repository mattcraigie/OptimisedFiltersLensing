import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import healpy as hp
import os
import pandas as pd
from .training import batch_apply


# Data handling functions and class

class GeneralDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class Scaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def norm_scale(x, axis=None):
    scaler = Scaler(np.mean(x, axis), np.std(x, axis))
    return scaler.transform(x), scaler


def data_shuffler(*args):
    size = args[0].shape[0]
    perm = torch.randperm(size)
    return tuple([arg[perm] for arg in args])


class DataHandler:
    """There are three types of data: patches, features and targets. They are all handled differently."""
    def __init__(self, load_subset=-1, sub_batch_subset=-1, val_ratio=0.2, test_ratio=0.2):
        self.load_subset = load_subset
        self.sub_batch_subset = sub_batch_subset
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.data = None
        self.targets = None

        self.data_scaler = None
        self.targets_scaler = None

        self.dataset = None

    def load_patches(self, path):
        all_dirs = os.listdir(path)
        all_dirs = np.sort(all_dirs)

        patches = []
        for dir_ in all_dirs[:self.load_subset]:
            patches.append(np.load(os.path.join(path, dir_))[:self.sub_batch_subset])

        return np.stack(patches)

    def load_features(self, path):
        features = np.load(path)
        features = features[:self.load_subset, :self.sub_batch_subset]

        return features

    def add_data(self, path, patches=False, normalise=True, log=False):
        if patches:
            data = self.load_patches(path)
        else:
            data = self.load_features(path)

        if log:
            data = np.log(data)

        if normalise:
            data, self.data_scaler = norm_scale(data, axis=0)

        self.data = torch.from_numpy(data).float()

        if self.targets is not None:
            assert self.data.shape[0] == self.targets.shape[0], 'Data and targets must have same number of samples'
            self.data, self.targets = data_shuffler(self.data, self.targets)
            self.dataset = GeneralDataset(self.data, self.targets)


    def add_targets(self, path, normalise=True, use_params=('s8',)):
        df = pd.read_csv(path)
        targets = df[list(use_params)].values
        targets = targets[:self.load_subset]

        if normalise:
            targets, self.targets_scaler = norm_scale(targets, axis=0)

        self.targets = torch.from_numpy(targets).float()

        if self.data is not None:
            assert self.data.shape[0] == self.targets.shape[0], 'Data and targets must have same number of samples'
            self.data, self.targets = data_shuffler(self.data, self.targets)
            self.dataset = GeneralDataset(self.data, self.targets)

    def get_test_loader(self, batch_size=128):
        assert self.dataset is not None, 'Data and targets must be loaded before getting loaders'
        num_data = len(self.dataset)
        test_split = int(self.test_ratio * num_data)
        test_indices = np.arange(test_split)
        test_sampler = SubsetRandomSampler(test_indices)
        return DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler)

    def get_train_val_loaders(self, subset=-1, batch_size=128):

        # make the samplers
        num_data = len(self.dataset)
        test_split = int(self.test_ratio * num_data)
        remaining = int((1 - self.test_ratio) * num_data)
        remaining = remaining if subset == -1 else subset

        val_split = int(self.val_ratio * remaining)

        val_indices = np.arange(test_split, test_split + val_split)
        train_indices = np.arange(test_split + val_split, test_split + val_split + remaining)

        val_sampler = SubsetRandomSampler(val_indices)
        train_sampler = SubsetRandomSampler(train_indices)

        val_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=val_sampler)
        train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)

        return train_loader, val_loader


# Patch making functions

def compute_patch_centres(patch_nside, mask=None, threshold=0.2):
    # compute patch centres by using a smaller healpix map with patch_nside, and taking midpoints from the larger pixels
    patch_map = np.arange(hp.nside2npix(patch_nside))
    patch_thetas, patch_phis = hp.pix2ang(patch_nside, patch_map)  # theta and phi are the centrepoints of the patches

    patch_thetas = np.rad2deg(patch_thetas)
    patch_phis = np.rad2deg(patch_phis)

    if mask is None:
        return [(i, j) for i, j in zip(patch_thetas, patch_phis)]

    warnings.warn('Masking not tested yet')

    # Downside the mask to the smaller map size. ud_grade uses an averaging to downsample. As a result, the pixel value
    # in the smaller map will represent the number of pixels in the mask
    mask_downsized = hp.ud_grade(mask, patch_nside)

    patch_centres = []
    for patch_theta, patch_phi in zip(patch_thetas, patch_phis):
        if mask_downsized[hp.ang2pix(patch_nside, patch_theta, patch_phi)] > threshold:
            patch_centres.append((patch_theta, patch_phi))

    return patch_centres


def healpix_map_to_patches(healpix_map, patch_centres, patch_size, resolution):
    patch_set = []
    for patch_ra, patch_dec in patch_centres:
        # gnomview vs cartview? I think they are both equivalent for dtheta, dphi ~ 0
        gnomonic_projection = hp.gnomview(healpix_map,
                                          xsize=patch_size,
                                          rot=(patch_ra, patch_dec),
                                          no_plot=True,
                                          reso=resolution,
                                          return_projected_map=True)
        gnomonic_projection = gnomonic_projection.astype(np.half).filled()  # convert to normal array
        patch_set.append(gnomonic_projection)
    patch_set = np.stack(patch_set)
    return patch_set


# Other functions

def load_and_apply(load_path, save_path, function, device):

    all_dirs = os.listdir(os.path.join(load_path))
    all_dirs = np.sort(all_dirs)

    data = []
    for dir_ in all_dirs[:1000]:
        fields = torch.from_numpy(np.load(os.path.join(load_path, dir_)))
        results = batch_apply(fields, 64, function, device=device)
        data.append(results.cpu().numpy())
    data = np.stack(data)

    np.save(save_path, data)


