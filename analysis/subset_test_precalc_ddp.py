import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ostlensing.training import mse, Trainer
from ostlensing.dataloading import DataHandler, load_and_apply
from ostlensing.models import PreCalcRegressor
from datascaling_ddp import setup, cleanup, mse

from scattering_transform.scattering_transform import ScatteringTransform2d, Reducer
from scattering_transform.filters import Morlet, FixedFilterBank
from scattering_transform.power_spectrum import PowerSpectrum


# Pre-calculation functions

def pre_calc_mst():
    path = "//pscratch/sd/m/mcraigie/cosmogrid/"

    load_path = os.path.join(path, '/patches/')
    save_path = os.path.join(path, '/pre_calc_mst.npy')

    if not os.path.exists(save_path):
        size, num_scales, num_angles = 128, 4, 4
        morlet = Morlet(128, num_scales, num_angles)
        mst = ScatteringTransform2d(morlet)
        reducer = Reducer(morlet, reduction=None)
        load_and_apply(load_path, save_path, lambda x: reducer(mst(x)), device=torch.device('cuda:0'))


def pre_calc_ost():
    path = "//pscratch/sd/m/mcraigie/cosmogrid/"

    filter_name = 'name'
    filter_path = os.path.join(path, '/saved_filters/{}.pyt'.format(filter_name))
    load_path = os.path.join(path, '/patches/')
    save_path = os.path.join(path, '/pre_calc_ost_{}.npy'.format(filter_name))

    if not os.path.exists(save_path):
        filters = torch.load(filter_path)
        filter_bank = FixedFilterBank(filters)
        ost = ScatteringTransform2d(filter_bank)
        reducer = Reducer(filter_bank, reduction=None)
        load_and_apply(load_path, save_path, lambda x: reducer(ost(x)), device=torch.device('cuda:0'))


def pre_calc_pk():
    path = "//pscratch/sd/m/mcraigie/cosmogrid/"

    load_path = os.path.join(path, '/patches/')
    save_path = os.path.join(path, '/pre_calc_pk.npy')

    if not os.path.exists(save_path):
        power_spectrum = PowerSpectrum(size=128, num_bins=20)
        load_and_apply(load_path, save_path, lambda x: power_spectrum(x), device=torch.device('cuda:0'))


# Analysis functions

