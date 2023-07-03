import torch
import argparse
import yaml
import os
import numpy as np



from scattering_transform.scattering_transform import ScatteringTransform2d, Reducer
from scattering_transform.filters import Morlet, FixedFilterBank
from scattering_transform.power_spectrum import PowerSpectrum

from ostlensing.dataloading import load_and_apply, Scaler
from ostlensing.models import ResNetWrapper


def st_func(filters, reduction, device):
    st = ScatteringTransform2d(filters)
    st.to(device)
    reducer = Reducer(filters, reduction=reduction)
    return lambda x: reducer(st(x))


def mst(size, num_scales, num_angles, reduction, device):
    morlet = Morlet(size, num_scales, num_angles)
    return st_func(morlet, reduction, device)


def ost(filter_path, reduction, device):
    filters = torch.load(filter_path)
    filter_bank = FixedFilterBank(filters)
    return st_func(filter_bank, reduction, device)


def resnet(model_state_dict_path, pretrained_model, device):
    model = ResNetWrapper(pretrained_model=pretrained_model)
    model = model.load_state_dict(model_state_dict_path)
    model.to(device)
    return model


def pk(size, num_bins, device):
    ps = PowerSpectrum(size, num_bins)
    ps.to(device)
    return ps


def pre_calc(load_path, save_path, file_name, method, kwargs):
    function_mapping = {'ost': ost, 'mst': mst, 'ps': pk, 'resnet': resnet}
    load_and_apply(load_path, function_mapping[method](**kwargs), device=torch.device('cuda:0'),
                   save_path=os.path.join(save_path, file_name + '_full.npy'))


def subset_average_standardise(save_path, file_name, subsets):
    data = np.load(os.path.join(save_path, file_name + '_full.npy'))

    for subset in subsets:
        if subset is not None:
            subset_data = data[:, :subset].mean(axis=1)

            main_path = os.path.join(save_path, f'{file_name}_{str(subset)}')

            # without altering data
            np.save(main_path + '.npy', subset_data)

            # with std
            scaler = Scaler(np.mean(subset_data), np.std(subset_data))
            std_data = scaler.transform(subset_data)
            np.save(main_path + '_std.npy', std_data)

            # with std and log
            log_data = np.log(subset_data)
            scaler = Scaler(np.mean(log_data), np.std(log_data))
            log_std_data = scaler.transform(log_data)
            np.save(main_path + '_log_std.npy', log_std_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/precalc/ps.yml', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    load_path = config['load_path']
    save_path = config['save_path']
    file_name = config['file_name']
    method = config['method']
    kwargs = config['kwargs']

    already_run = config['already_run']
    subsets = config['subsets']

    if not already_run:
        pre_calc(load_path, save_path, file_name, method, kwargs)
    subset_average_standardise(save_path, file_name, subsets)

if __name__ == '__main__':
    main()