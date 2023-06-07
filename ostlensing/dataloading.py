import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import healpy as hp


class GeneralDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def data_preprocessing_cosmogrid(test=False):
    # note that our data will come in the form (num_data, sub_batch, size, size)
    # where sub_batch is all the fields from the same cosmology

    if not test:
        torch.manual_seed(0)
        num = 1000
    else:
        torch.manual_seed(1)
        num = 100

    sub_num = 10
    sigs = torch.rand(num, 1, 1, 1)
    data = torch.normal(0, 1, size=(num, sub_num, 32, 32)) * sigs
    targets = sigs[:, :, 0, 0]
    return data, targets


def data_preprocessing_dirac():
    # Data from the Dirac sims
    return data_preprocessing_cosmogrid(test=True)


def make_dataloaders(data, targets, batch_size=8, seed=42):
    dataset = GeneralDataset(data, targets)

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
        gnomonic_projection = gnomonic_projection.astype(np.half).compressed()  # convert to normal array
        patch_set.append(gnomonic_projection)
    patch_set = np.stack(patch_set)
    return patch_set
