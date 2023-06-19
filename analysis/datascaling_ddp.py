import numpy as np
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ostlensing.training import mse_and_admissibility, Trainer
from ostlensing.dataloading import DataHandler
from ostlensing.models import OptimisableSTRegressor, PreCalcRegressor

from ddp_nersc import ddp_main, setup, cleanup

# help options: berkeley group desi -  nersc channel desi anthony kremin,

model_map = {'ost': OptimisableSTRegressor, 'pre_calc': PreCalcRegressor}


# Adjusted loss functions

def mse_and_admissibility_ddp(output, target, model, weighting=1.0):
    return mse_and_admissibility(output, target, model.module, weighting)


def mse(output, target, model):
    return nn.functional.mse_loss(output, target)


# main analysis function

def data_scaling(rank, args):
    """Run the data scaling analysis. Outputs the results to a csv file.

    Args:
        rank (int): The rank of the process.
        args (argparse.Namespace): The command line arguments. This function only uses args.config.
    """

    print(f"Running data scaling analysis on rank {rank}.")
    setup(rank, args)

    # Load configuration settings
    config = args.config

    # data params
    data_config = config['data']
    data_path = data_config['data_path']
    data_subpath = data_config['data_subpath']  # i.e. patches or pre_calc
    is_patches = data_config['is_patches']
    load_subset = data_config['load_subset']
    sub_batch_subset = data_config['sub_batch_subset']
    val_ratio = data_config['val_ratio']
    test_ratio = data_config['test_ratio']

    # model params
    model_config = config['model']
    model_type = model_config['model_type']
    submodel_type = model_config['submodel_type']
    model_args = model_config['model_args']

    # training params
    train_config = config['training']
    num_epochs = train_config['num_epochs']
    batch_size = train_config['batch_size']
    learning_rate = train_config['learning_rate']

    # analysis params
    analysis_config = config['analysis']
    data_subsets = analysis_config['data_subsets']

    # make output folder
    if submodel_type is not None:
        out_folder = os.path.join('outputs', 'datascaling', model_type, submodel_type)
    else:
        out_folder = os.path.join('outputs', 'datascaling', model_type)

    if rank == 0:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # load train+val and test data with DataHandler
    data_handler = DataHandler(load_subset=load_subset,
                               sub_batch_subset=sub_batch_subset,
                               val_ratio=val_ratio,
                               test_ratio=test_ratio)
    print("ddp - adding data")
    data_handler.add_data(os.path.join(data_path, data_subpath), patches=is_patches, normalise=True, log=True)

    print("ddp - adding targets")
    data_handler.add_targets(os.path.join(data_path, 'params.csv'), normalise=True)

    print("ddp - making test loader")
    # make test loader outside the loop
    test_loader = data_handler.get_test_loader()

    # setup train and test losses
    train_criterion = mse_and_admissibility_ddp if model_type == 'ost' else mse
    test_criterion = mse

    # make this a proper outputs -- make folder etc.
    model_results = []
    for subset in data_subsets:
        if rank == 0:
            print(f"Running analysis for data subset {subset}.")

        # make train and val loaders with the subset of data
        train_loader, val_loader = data_handler.get_train_val_loaders(subset=subset, batch_size=batch_size)

        # set up the model
        try:
            model_class = model_map[model_type]
        except KeyError:
            raise ValueError('Model type not recognised.')

        try:
            model = model_class(**model_args)
        except ValueError:
            raise ValueError('Model arguments not recognised.')

        # send to gpu and wrap in DDP
        model.to(rank)
        model = DDP(model, device_ids=[rank])

        # set up the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # train the model using Trainer
        trainer = Trainer(model, optimizer, train_criterion, test_criterion, train_loader, val_loader, test_loader,
                          rank, ddp=True)
        trainer.train_loop(num_epochs)

        # test the best validated model with unseen data
        test_loss = trainer.test(load_best=True)  # reduced across ranks and stored in rank 0's test_loss

        # only save results for rank 0
        if rank == 0:
            subset_folder = os.path.join(out_folder, f'subset_{subset}')
            if not os.path.exists(subset_folder):
                os.makedirs(subset_folder)
            trainer.save_model(os.path.join(subset_folder, 'model.pt'))
            trainer.save_losses(os.path.join(subset_folder, 'losses.pt'))
            trainer.save_predictions(os.path.join(subset_folder, 'predictions.pt'))  # could speed this up by sharing across ranks
            trainer.save_targets(os.path.join(subset_folder, 'targets.pt'))
            model_results.append(test_loss.cpu().item())

    if rank == 0:  # only save the results once!
        df = pd.DataFrame({'data_subset': data_subsets, 'test_loss': model_results})
        df.to_csv(os.path.join(out_folder, 'data_scaling_{}.csv'.format(model_type)), index=False)

    cleanup()


if __name__ == '__main__':
    print("starting...")
    ddp_main(data_scaling)
    print("finished!")