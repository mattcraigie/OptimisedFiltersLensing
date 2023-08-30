import numpy as np
import os
import pandas as pd
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ostlensing.training import mse_and_admissibility, Trainer, mse
from ostlensing.dataloading import DataHandler
from ostlensing.models import ModelRegressor

from ddp_nersc import ddp_main, setup, cleanup
import torch.distributed as dist


# Create a custom filter to suppress specific log messages
class SuppressFilter(logging.Filter):
    def filter(self, record):
        # Suppress log messages from pytorch (because they cram my logging!)
        return "Reducer buckets have been rebuilt" not in record.getMessage()


# Adjusted loss functions

def mse_and_admissibility_ddp(output, target, regressor, weighting=1.0):
    return mse_and_admissibility(output, target, regressor.module.model, weighting)


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
    pretrained_model = regressor_config['pretrained_model']

    # training params
    train_config = config['training']
    num_epochs = train_config['num_epochs']
    batch_size = train_config['batch_size']
    learning_rates = train_config['learning_rate']


    # analysis params
    analysis_config = config['analysis']
    data_subsets = analysis_config['data_subsets']
    repeats = analysis_config['repeats']
    analysis_name = analysis_config['analysis_name']

    if not isinstance(learning_rates, list):
        learning_rates = [learning_rates] * len(data_subsets)

    if not isinstance(num_epochs, list):
        num_epochs = [num_epochs] * len(data_subsets)

    # set up logging
    logging_filename = os.path.join('outputs', 'logs', f'{data_type}_{model_type}_{analysis_name}.log')
    print(logging_filename)
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
    data_handler = DataHandler(**datahandler_kwargs, rank=rank, world_size=dist.get_world_size())

    data_handler.add_data(os.path.join(data_path, data_type, data_subpath), patches=data_type == 'patches', normalise=False,
                          log=False)

    # use_params = ('s8',)
    # use_params = ('Om',)
    # use_params = ('s8', 'As', 'bary_Mc', 'bary_nu', 'H0', 'O_cdm', 'O_nu', 'Ob', 'Om', 'ns', 'w0')
    # use_params = ('s8', 'As', 'O_cdm', 'Om')
    use_params = ('s8', 'Om')
    regressor_kwargs['regressor_outputs'] = len(use_params)

    data_handler.add_targets(os.path.join(data_path, 'params_std.csv'), normalise=False,
                             use_params=use_params)

    # make test loader outside the loop for consistent test data
    test_loader = data_handler.get_test_loader(ddp=True)

    # setup train and test losses
    # train_criterion = mse_and_admissibility_ddp if model_type == 'ost' and data_type == 'patches' else mse
    train_criterion = mse
    test_criterion = mse

    # iterate of repeat number i
    for i in range(repeats):
        if rank == 0:
            repeat_start_time = time.time()
            logging.info(f"\nRunning repeat {i}.")

        # set the seed for the train and val split. This should be consistent amongst the subsets for the same repeat
        data_handler.seed = i

        # iterate over the train/val data subsets
        for subset, learning_rate, num_epoch in zip(data_subsets, learning_rates, num_epochs):
            if rank == 0:
                subset_start_time = time.time()
                logging.info(f"Running subset {subset}.")

            logging.debug(f"Making train and val loaders on rank {rank}")
            train_loader, val_loader = data_handler.get_train_val_loaders(subset=subset, batch_size=batch_size, ddp=True)

            logging.debug(f"Setting up the regressor on rank {rank}")
            regressor = ModelRegressor(**regressor_kwargs)

            try:
                regressor.load_state_dict(torch.load(pretrained_model))
                if rank == 0:
                    logging.info("Pre-trained model loaded")
            except:
                if rank == 0:
                    logging.info("No pre-trained model loaded")

            logging.debug(f"Sending the regressor to rank {rank}")
            regressor.to(rank)
            regressor = DDP(regressor, device_ids=[rank], find_unused_parameters=False)

            logging.debug(f"Setting up the optimizer on rank {rank}")
            optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)

            logging.debug(f"Setting up the trainer on rank {rank}")
            trainer = Trainer(regressor, optimizer, train_criterion, test_criterion, train_loader, val_loader, test_loader,
                              rank, ddp=True)

            logging.debug(f"Running train and validation trainer on rank {rank}")
            trainer.train_loop(num_epoch)

            logging.debug(f"Making predictions on rank {rank}")
            trainer.load_best_model()
            trainer.make_predictions()

            if rank == 0:
                logging.debug(f"Saving the results on rank {rank}")

                # save the model and predictions if it's the first repeat
                repeat_padded = str(i).zfill(2)
                subset_padded = str(subset).zfill(4)
                subset_folder = os.path.join(out_folder, f'repeat_{repeat_padded}', f'subset_{subset_padded}')
                if not os.path.exists(subset_folder):
                    os.makedirs(subset_folder)
                trainer.save_all(subset_folder)

                subset_end_time = time.time()
                logging.info("Subset {} took {:.2f} seconds.".format(subset, subset_end_time - subset_start_time))

        if rank == 0:
            repeat_end_time = time.time()
            logging.info("Repeat {} took {:.2f} seconds.".format(i, repeat_end_time - repeat_start_time))

    cleanup()

    if rank == 0:
        end_time = time.time()
        logging.info("Data scaling analysis took {:.2f} seconds.".format(end_time - start_time))


if __name__ == '__main__':
    ddp_main(data_scaling)