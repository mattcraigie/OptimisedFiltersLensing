import numpy as np
import os
import pandas as pd
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ostlensing.training import mse_and_admissibility, Trainer
from ostlensing.dataloading import DataHandler
from ostlensing.models import ModelRegressor

from ddp_nersc import ddp_main, setup, cleanup


# Create a custom filter to suppress specific log messages
class SuppressFilter(logging.Filter):
    def filter(self, record):
        # Suppress log messages from pytorch (because they cram my logging!)
        return "Reducer buckets have been rebuilt" not in record.getMessage()


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

    # look at path and
    setup(rank, args)

    # Load configuration settings
    config = args.config

    # data params
    data_config = config['data']
    data_path = data_config['data_path']
    data_subpath = data_config['data_subpath']
    data_type = data_config['data_type']
    datahandler_kwargs = data_config['datahandler_kwargs']

    # regressor params
    regressor_config = config['regressor']
    regressor_kwargs = regressor_config['regressor_kwargs']
    model_type = regressor_kwargs['model_type']

    # training params
    train_config = config['training']
    num_epochs = train_config['num_epochs']
    batch_size = train_config['batch_size']
    learning_rate = train_config['learning_rate']

    # analysis params
    analysis_config = config['analysis']
    data_subsets = analysis_config['data_subsets']
    repeats = analysis_config['repeats']
    analysis_name = analysis_config['analysis_name']

    # set up logging
    logging_filename = os.path.join('outputs', 'logs', f'{data_type}_{model_type}.log')
    if rank == 0 and os.path.exists(logging_filename):
        os.remove(logging_filename)

    logging.basicConfig(filename=logging_filename, level=logging.INFO)
    logging.info(f"Running data scaling analysis on rank {rank}.")
    logger = logging.getLogger()
    logger.addFilter(SuppressFilter())  # suppress log messages from pytorch

    # make output folder
    out_folder = os.path.join('outputs', 'datascaling', data_type, model_type, analysis_name)

    if rank == 0:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        start_time = time.time()
        logging.info('Loading and initialising.')

    # load train+val and test data with DataHandler
    data_handler = DataHandler(**datahandler_kwargs)

    data_handler.add_data(os.path.join(data_path, data_type, data_subpath), patches=data_type == 'patches', normalise=False,
                          log=False)
    data_handler.add_targets(os.path.join(data_path, 'params_std.csv'), normalise=False, use_params=('s8',))

    # make test loader outside the loop for consistent test data
    test_loader = data_handler.get_test_loader()

    # setup train and test losses
    train_criterion = mse_and_admissibility_ddp if model_type == 'ost' else mse
    test_criterion = mse

    # set up the results dataframe
    df = pd.DataFrame({'data_subset': data_subsets})

    # iterate of repeat number i
    for i in range(repeats):
        if rank == 0:
            repeat_start_time = time.time()
            logging.info(f"\nRunning repeat {i}.")

        # set the seed for the train and val split. This should be consistent amongst the subsets for the same repeat
        data_handler.seed = i

        model_results = []

        # iterate over the train/val data subsets
        for subset in data_subsets:
            if rank == 0:
                subset_start_time = time.time()
                logging.info(f"Running subset {subset}.")

            # make train and val loaders with the subset of data
            train_loader, val_loader = data_handler.get_train_val_loaders(subset=subset, batch_size=batch_size)

            # set up the model
            regressor = ModelRegressor(**regressor_kwargs)

            # send to gpu and wrap in DDP
            regressor.to(rank)
            regressor = DDP(regressor, device_ids=[rank])

            # set up the optimizer
            optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)

            # train the model using Trainer
            trainer = Trainer(regressor, optimizer, train_criterion, test_criterion, train_loader, val_loader, test_loader,
                              rank, ddp=True)
            trainer.train_loop(num_epochs)

            # test the best validated model with unseen data
            test_loss = trainer.test(load_best=True)  # reduced across ranks and stored in rank 0's test_loss

            # only save results for rank 0
            if rank == 0:
                subset_folder = os.path.join(out_folder, f'subset_{subset}')
                if not os.path.exists(subset_folder):
                    os.makedirs(subset_folder)
                trainer.save_all(subset_folder)
                model_results.append(test_loss.cpu().item())

                subset_end_time = time.time()
                logging.info("Subset {} took {:.2f} seconds.".format(subset, subset_end_time - subset_start_time))

        if rank == 0:
            df[f'run_{str(i)}'] = model_results

            repeat_end_time = time.time()
            logging.info("Repeat {} took {:.2f} seconds.".format(i, repeat_end_time - repeat_start_time))

    if rank == 0:  # only save the results once!
        df.to_csv(os.path.join(out_folder, 'data_scaling_{}.csv'.format(model_type)), index=False)

    cleanup()

    if rank == 0:
        end_time = time.time()
        logging.info("Data scaling analysis took {:.2f} seconds.".format(end_time - start_time))


if __name__ == '__main__':
    ddp_main(data_scaling)
