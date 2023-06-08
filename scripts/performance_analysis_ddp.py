from ostlensing.training import mse_and_admissibility, train, validate
from ostlensing.dataloading import DataHandler
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
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

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

def train_loop(model, optimizer, train_criterion, val_criterion, train_loader, val_loader, device, epochs):
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_model_params = None

    for epoch in range(1, epochs + 1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
        train_loss = train(model, optimizer, train_criterion, train_loader, device)
        val_loss = validate(model, val_criterion, val_loader, device)

        if device == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_params = model.module.state_dict()  # modified

    return train_losses, val_losses, best_model_params


def test_performance(rank, world_size):
    print(f"Running OST DDP analysis on rank {rank}.")
    setup(rank, world_size)
    path = "//pscratch/sd/m/mcraigie/cosmogrid/"

    num_epochs = 100
    data_subsets = [100, 500, 1000]
    model_name = 'ost'

    # load train+val and test data
    data_handler = DataHandler(load_subset=1000, val_ratio=0.2, test_ratio=0.2)
    data_handler.add_data(os.path.join(path, '/patches/'), patches=True, normalise=True, log=True)
    data_handler.add_targets(os.path.join(path, 'params.csv'), normalise=True)

    # make test loader outside the loop
    test_loader = data_handler.get_test_loader()

    # setup train and test losses
    train_criterion = mse_and_admissibility_ddp
    test_criterion = mse

    model_results = []
    for subset in data_subsets:
        # make train and val loaders with the subset of data
        train_loader, val_loader = data_handler.get_train_val_loaders(subset=subset)

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

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # train the model
        train_losses, val_losses, best_model_params = train_loop(ddp_model,
                                                                 optimizer,
                                                                 train_criterion,
                                                                 test_criterion,
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
