import warnings
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import healpy as hp
import os
import pandas as pd
from .training import batch_apply


# Data handling functions and class


class Scaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class GeneralDataset(TensorDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)
        self.data = data
        self.targets = targets


def norm_scale(x, axis=None):
    scaler = Scaler(np.mean(x, axis), np.std(x, axis))
    return scaler.transform(x), scaler


def data_shuffler(*args, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    size = args[0].shape[0]
    perm = torch.randperm(size)
    return tuple([arg[perm] for arg in args])


class DataHandler:
    """There are three types of data: patches, features and targets. They are all handled differently."""

    def __init__(self, load_subset=None, patch_subset=None, val_ratio=0.2, test_ratio=0.2, seed=None, pre_average=False,
                 rank=None, world_size=None):
        self.load_subset = load_subset
        self.patch_subset = patch_subset
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.pre_average = pre_average  # pre average over the patch dimension
        self.rank = rank
        self.world_size = world_size

        self.data = None
        self.targets = None

        self.data_scaler = None
        self.targets_scaler = None

        self.patch_perm = None

    def load_patches(self, path):
        all_dirs = [i for i in os.listdir(path) if not i.startswith('.')]
        all_dirs = np.sort(all_dirs)

        patches = []
        for i, dir_ in enumerate(all_dirs[:self.load_subset]):
            loaded_patches = np.load(os.path.join(path, dir_))
            num_patches = loaded_patches.shape[0]
            np.random.seed(self.seed + i)  # different for every cosmology, same for every separate analysis
            patches.append(loaded_patches[np.random.permutation(num_patches)][:self.patch_subset])

        return np.stack(patches)

    def load_features(self, path):
        features = np.load(path)
        features = features[:self.load_subset]

        features_subset = np.zeros((self.load_subset, self.patch_subset, features.shape[-1]))

        for i in range(self.load_subset):
            np.random.seed(self.seed + i)  # different for every cosmology, same for every separate analysis
            features_subset[i] = features[i][np.random.permutation(features.shape[1])[:self.patch_subset]]

        if self.pre_average:
            features_subset = features_subset.mean(axis=1)

        return features_subset

    def add_data(self, path, patches=False, normalise=False, log=False):
        if patches:
            data = self.load_patches(path)
        else:
            data = self.load_features(path)

        # I think I should pre-normalise and pre-log the data
        if log:
            data = np.log(data)

        if normalise:
            data, self.data_scaler = norm_scale(data, axis=0)

        self.data = torch.from_numpy(data).float()

        if self.targets is not None:
            assert self.data.shape[0] == self.targets.shape[0], 'Data and targets must have same number of samples'
            self.data, self.targets = data_shuffler(self.data, self.targets, seed=self.seed)

    def add_targets(self, path, normalise=False, use_params=None):

        df = pd.read_csv(path)

        if use_params is None:
            use_params = df.columns[1:]

        targets = df[list(use_params)].values
        targets = targets[:self.load_subset]

        if normalise:
            targets, self.targets_scaler = norm_scale(targets, axis=0)

        self.targets = torch.from_numpy(targets).float()
        if self.data is not None:
            assert self.data.shape[0] == self.targets.shape[0], 'Data and targets must have same number of samples'
            self.data, self.targets = data_shuffler(self.data, self.targets, seed=self.seed)


    def get_test_loader(self, batch_size=128, ddp=False):
        assert self.data is not None and self.targets is not None, \
            'Data and targets must be loaded before getting dataloaders'
        num_data = len(self.data)
        test_split = int(self.test_ratio * num_data)
        test_dataset = GeneralDataset(self.data[:test_split], self.targets[:test_split])
        if ddp:
            test_sampler = DistributedSampler(test_dataset, drop_last=True, shuffle=False)
            return DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0)
        else:
            return DataLoader(test_dataset, batch_size=batch_size)

    def get_train_val_loaders(self, subset=None, batch_size=128, ddp=False):
        assert self.data is not None and self.targets is not None, \
            'Data and targets must be loaded before getting dataloaders'

        # make the samplers
        num_data = len(self.data)
        test_split = int(self.test_ratio * num_data)  # test set is unaffected by the subsetting
        num_remaining = num_data - test_split

        if subset > num_remaining:
            raise ValueError(
                f"Subset must be smaller than or equal to the non-test data (non test data: {num_remaining}). "
                f"Load more data or adjust subset.")
        num_remaining = num_remaining if subset is None else subset

        # another layer of randomness to get the bootstrapping working between repeats
        leftover_data = self.data[test_split:num_remaining+test_split]
        leftover_targets = self.targets[test_split:num_remaining+test_split]
        leftover_data, leftover_targets = data_shuffler(leftover_data, leftover_targets, seed=self.seed)

        val_split = int(self.val_ratio * num_remaining)

        train_dataset = GeneralDataset(leftover_data[val_split:],
                                       leftover_targets[val_split:])
        val_dataset = GeneralDataset(leftover_data[:val_split],
                                     leftover_targets[:val_split])


        if ddp:
            train_sampler = DistributedSampler(train_dataset, drop_last=True, shuffle=True, seed=self.seed)
            val_sampler = DistributedSampler(val_dataset, drop_last=True, shuffle=True, seed=self.seed)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)


        return train_loader, val_loader




# Patch making functions

def compute_patch_centres(patch_nside, mask=None, threshold=0.2):
    # compute patch centres by using a smaller healpix map with patch_nside, and taking midpoints from the larger pixels
    patch_map = np.arange(hp.nside2npix(patch_nside))
    patch_thetas, patch_phis = hp.pix2ang(patch_nside, patch_map)  # theta and phi are the centrepoints of the patches



    if mask is None:
        patch_thetas = np.rad2deg(patch_thetas)
        patch_phis = np.rad2deg(patch_phis)
        return [(i, j) for i, j in zip(patch_thetas, patch_phis)]

    # Downside the mask to the smaller map size. ud_grade uses an averaging to downsample. As a result, the pixel value
    # in the smaller map will represent the number of pixels in the mask
    mask_downsized = hp.ud_grade(mask, patch_nside)

    patch_centres = []
    for patch_theta, patch_phi in zip(patch_thetas, patch_phis):
        if mask_downsized[hp.ang2pix(patch_nside, patch_theta, patch_phi)] > threshold:
            patch_theta = np.rad2deg(patch_theta)
            patch_phi = np.rad2deg(patch_phi)
            patch_centres.append((patch_theta, patch_phi))

    return patch_centres


def healpix_map_to_patches(healpix_map, patch_centres, patch_size, resolution):
    patch_set = []
    for patch_ra, patch_dec in patch_centres:
        rot= (patch_dec, -patch_ra + 90)
        # gnomview vs cartview? I think they are both equivalent for dtheta, dphi ~ 0
        gnomonic_projection = hp.gnomview(healpix_map,
                                          xsize=patch_size,
                                          rot=rot,
                                          no_plot=True,
                                          reso=resolution,
                                          return_projected_map=True)
        gnomonic_projection = gnomonic_projection.astype(np.half).filled()  # convert to normal array
        patch_set.append(gnomonic_projection)
    patch_set = np.stack(patch_set)
    return patch_set


# Other functions

def load_and_apply(load_path, function, device, save_path=None):
    all_dirs = os.listdir(os.path.join(load_path))
    all_dirs = np.sort(all_dirs)

    data = []
    for dir_ in all_dirs:
        fields = torch.from_numpy(np.load(os.path.join(load_path, dir_))).float()
        results = batch_apply(fields, 16, function, device=device)
        data.append(results.cpu().numpy())

    data = np.stack(data)

    if save_path is None:
        return data
    else:
        np.save(save_path, data)




