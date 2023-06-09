from ostlensing.training import mse_and_admissibility, mse, train_loop, validate
from ostlensing.dataloading import DataHandler
from ostlensing.models import OptimisableSTRegressor
import numpy as np

import os
import torch
import torch.nn as nn
import torch.optim as optim


# Analysis functions

def test_performance(device):
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
    train_criterion = mse_and_admissibility
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
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # train the model
        train_losses, val_losses, best_model_params = train_loop(model,
                                                                 optimizer,
                                                                 train_criterion,
                                                                 test_criterion,
                                                                 train_loader,
                                                                 val_loader,
                                                                 device,
                                                                 num_epochs)

        # test the best validated model with unseen data. Only test on one GPU.

        model.load_state_dict(best_model_params)
        with torch.no_grad():
            test_loss = validate(model, test_criterion, test_loader, device)
            model_results.append(test_loss)


    model_results = np.array(model_results)
    np.save('model_{}_performance.npy'.format(model_name), model_results)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_performance(device)
