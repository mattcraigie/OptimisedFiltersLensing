import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import healpy as hp
import os
import pandas as pd

class GeneralDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class Scaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def data_preprocessing(path, test=False):
    # # use os to list all files in the directory, and load them one by one and stack them together

    patch_path = 'patches'
    targets_path = 'params.csv'

    data = []
    all_dirs = os.listdir(os.path.join(path, patch_path))
    all_dirs = np.sort(all_dirs)

    for dir_ in all_dirs:
        data.append(np.load(os.path.join(path, dir_)))

    df = pd.read_csv(targets_path)
    use_params = ['s8']
    # use_params = ['As', 'bary_Mc', 'bary_nu', 'H0', 'O_cdm', 'O_nu', 'Ob', 'Ol', 'Om', 'm_nu', 'ns', 's8', 'w0']

    targets = df[use_params].values

    data = torch.from_numpy(data).float().log()
    data_scaler = Scaler(data.mean(), data.std())
    data = data_scaler.transform(data)

    targets = torch.from_numpy(targets).float()
    target_scaler = Scaler(targets.mean(0), targets.std(0))
    targets = target_scaler.transform(targets)

    test_fraction = 0.2
    num_data = len(data)
    indices = list(range(num_data))
    split = int(np.floor(test_fraction * num_data))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    if test:
        return data[test_indices], targets[test_indices]
    else:
        return data[train_indices], targets[train_indices], data_scaler, target_scaler


def make_dataloaders(data, targets, batch_size=8, seed=42, test=False):
    dataset = GeneralDataset(data, targets)
    if test:
        return DataLoader(dataset, batch_size=batch_size)

    # randomly split data into train and validation sets
    np.random.seed(seed)
    num_data = len(dataset)
    indices = list(range(num_data))
    split = int(np.floor(0.2 * num_data))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)


    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


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
