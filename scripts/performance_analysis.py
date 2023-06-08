from ostlensing.training import train_loop, batch_apply, mse_and_admissibility
from ostlensing.dataloading import make_dataloaders, data_preprocessing
from ostlensing.ostmodel import OptimisableSTRegressor
import torch
from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def full_train(data, targets, model, num_epochs):
    train_loader, val_loader = make_dataloaders(data, targets)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = mse_and_admissibility

    return train_loop(model,
                      optimizer,
                      criterion,
                      train_loader,
                      val_loader,
                      device,
                      num_epochs)


def full_test(data, targets, model):
    with torch.no_grad():
        predictions = batch_apply(data, 64, model, device).cpu()
        return nn.functional.mse_loss(predictions, targets).item()


def show_performance(path):
    results = np.load(path)
    plt.plot(results)
    plt.show()


def main():
    torch.manual_seed(0)

    num_epochs = 10
    data_amounts = range(100, 500, 100)
    model_name = 'ost'

    data, targets = data_preprocessing_cosmogrid()
    data_test, targets_test = data_preprocessing_cosmogrid(test=True)

    model_results = []
    for amount in data_amounts:
        print(amount)
        data_subset, targets_subset = data[:amount], targets[:amount]
        model = OptimisableSTRegressor(size=32,
                                       num_scales=4,
                                       num_angles=4,
                                       reduction=None,
                                       hidden_sizes=(32, 32),
                                       output_size=1,
                                       activation=nn.LeakyReLU,
                                       seed=0)
        model.to(device)
        train_loss, val_loss, model_params, filters = full_train(data_subset, targets_subset, model, num_epochs)
        model.load_state_dict(model_params)
        test_score = full_test(data_test, targets_test, model)
        model_results.append(test_score)

    model_results = np.array(model_results)
    np.save('model_{}_performance.npy'.format(model_name), model_results)

    show_performance('model_{}_performance.npy'.format(model_name))


if __name__ == '__main__':
    main()
