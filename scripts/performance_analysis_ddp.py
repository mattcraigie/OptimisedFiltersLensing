from ostlensing.training import mse_and_admissibility, train, validate
from ostlensing.dataloading import make_dataloaders, data_preprocessing
from ostlensing.ostmodel import OptimisableSTRegressor
import numpy as np

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


# DDP functions

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# Adjusted loss functions

def mse_and_admissibility_ddp(output, target, model, weighting=1.0):
    return mse_and_admissibility(output, target, model.module, weighting)


def mse(output, target, model):
    return nn.functional.mse_loss(output, target)


# Analysis functions

def train_loop(model, optimizer, criterion, train_loader, val_loader, device, epochs=10):
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_model_params = None
    best_filters = None

    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        val_loss = validate(model, criterion, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_params = model.module.state_dict()  # modified
            best_filters = model.module.filters.filter_tensor  # modified

    return train_losses, val_losses, best_model_params, best_filters


def test_performance(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    path = "//pscratch/sd/m/mcraigie/cosmogrid/"

    num_epochs = 500
    data_amounts = range(10, 105, 5)
    model_name = 'ost'

    # load train+val and test data
    data, targets = data_preprocessing(path)
    data_test, targets_test = data_preprocessing(path, test=True)

    # make test loader outside the loop
    test_loader = make_dataloaders(data_test, targets_test, test=True)

    # setup train and test losses
    train_criterion = mse_and_admissibility_ddp
    test_criterion = mse

    model_results = []
    for amount in data_amounts:

        # trim to data subset and make train and val loaders
        print(amount)
        data_subset, targets_subset = data[:amount], targets[:amount]
        train_loader, val_loader = make_dataloaders(data_subset, targets_subset)

        # set up the model, send it to DDP and train
        model = OptimisableSTRegressor(size=32,
                                       num_scales=4,
                                       num_angles=4,
                                       reduction=None,
                                       hidden_sizes=(32, 32),
                                       output_size=1,
                                       activation=nn.LeakyReLU,
                                       seed=0)
        model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # train the model
        train_losses, val_losses, best_model_params, best_filters = train_loop(ddp_model,
                                                                               optimizer,
                                                                               train_criterion,
                                                                               train_loader,
                                                                               val_loader,
                                                                               rank,
                                                                               num_epochs)

        # test the best validated model with unseen data. Only test on one GPU.
        if rank == 0:
            model.load_state_dict(best_model_params)
            with torch.no_grad():
                test_loss = validate(model, test_criterion, test_loader, rank)
                model_results.append(test_loss)

    if rank == 0:  # only save the results once!
        model_results = np.array(model_results)
        np.save('model_{}_performance.npy'.format(model_name), model_results)

    cleanup()


def main(func, world_size):
    mp.spawn(func,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main(test_performance, world_size=4)
