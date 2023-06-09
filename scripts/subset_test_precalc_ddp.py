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
from subset_test_ost_ddp import setup, cleanup, mse

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


def test_performance(rank, world_size):
    print(f"Running pre-calced DDP on rank {rank}.")
    setup(rank, world_size)
    path = "//pscratch/sd/m/mcraigie/cosmogrid/"

    num_epochs = 100
    data_subsets = [100, 500, 1000]
    model_name = 'mst'

    # load train+val and test data
    data_handler = DataHandler(load_subset=1000, val_ratio=0.2, test_ratio=0.2)
    data_handler.add_data(os.path.join(path, '/pre_calc_mst.npy'), patches=False, normalise=True, log=False)
    data_handler.add_targets(os.path.join(path, 'params.csv'), normalise=True)

    # make test loader outside the loop
    test_loader = data_handler.get_test_loader()

    # setup train and test losses
    train_criterion = mse
    test_criterion = mse

    model_results = []
    for subset in data_subsets:
        # make train and val loaders with the subset of data
        train_loader, val_loader = data_handler.get_train_val_loaders(subset=subset)

        # set up the model, send it to DDP and train
        model = PreCalcRegressor(data_handler.data.shape[-1],
                                 hidden_sizes=(32, 32, 32),
                                 output_size=1,
                                 activation=nn.LeakyReLU,
                                 seed=0)
        model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # train the model
        trainer = Trainer(ddp_model, optimizer, train_criterion, test_criterion, train_loader, val_loader, rank,
                          distributed=True)
        trainer.train_loop(num_epochs)

        # test the best validated model with unseen data. Only test on one GPU.
        if rank == 0:
            test_loss = trainer.test(test_loader, load_best=True)
            model_results.append(test_loss)

    if rank == 0:  # only save the results once!
        model_results = np.array(model_results)
        np.save('model_{}_performance.npy'.format(model_name), model_results)

    cleanup()


def main(func, world_size):
    pre_calc_mst()
    mp.spawn(func,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main(test_performance, world_size=4)
