from ostlensing.training import batch_apply, mse_and_admissibility, train, validate
from ostlensing.dataloading import make_dataloaders
from ostlensing.ostmodel import OptimisableSTRegressor
import matplotlib.pyplot as plt
import numpy as np

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def mse_and_admissibility_ddp(output, target, model, weighting=1.0):
    return mse_and_admissibility(output, target, model.module, weighting)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
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
    data = torch.randn((100, 10, 32, 32))
    targets = torch.randn((100, 1))

    train_loader, val_loader = make_dataloaders(data, targets)

    num_epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = mse_and_admissibility_ddp

    train_losses, val_losses, best_model_params, best_filters = train_loop(model,
                                                                           optimizer,
                                                                           criterion,
                                                                           train_loader,
                                                                           val_loader,
                                                                           rank,
                                                                           num_epochs)

    cleanup()

    if rank == 0:
        plt.imshow(best_filters[0, 0].cpu().detach().numpy())
        plt.savefig('learned_filter.png')

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    # Specify the number of processes or workers
    world_size = 4

    # Run the DDP example
    run_demo(demo_basic, world_size)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#

#
#
# def full_test(data, targets, model):
#     with torch.no_grad():
#         predictions = batch_apply(data, 64, model, device).cpu()
#         return nn.functional.mse_loss(predictions, targets).item()
#
#
# def show_performance(path):
#     results = np.load(path)
#     plt.plot(results)
#     plt.show()
#
#
# def main():
#     torch.manual_seed(0)
#
#     num_epochs = 10
#     data_amounts = range(100, 500, 100)
#     model_name = 'ost'
#
#     data, targets = data_preprocessing_cosmogrid()
#     data_test, targets_test = data_preprocessing_cosmogrid(test=True)
#
#     model_results = []
#     for amount in data_amounts:
#         print(amount)
#         data_subset, targets_subset = data[:amount], targets[:amount]
#         model = OptimisableSTRegressor(size=32,
#                                        num_scales=4,
#                                        num_angles=4,
#                                        reduction=None,
#                                        hidden_sizes=(32, 32),
#                                        output_size=1,
#                                        activation=nn.LeakyReLU,
#                                        seed=0)
#         model.to(device)
#         train_loss, val_loss, model_params, filters = full_train(data_subset, targets_subset, model, num_epochs)
#         model.load_state_dict(model_params)
#         test_score = full_test(data_test, targets_test, model)
#         model_results.append(test_score)
#
#     model_results = np.array(model_results)
#     np.save('model_{}_performance.npy'.format(model_name), model_results)
#
#     show_performance('model_{}_performance.npy'.format(model_name))
#
#
# if __name__ == '__main__':
#     main()
