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


def pre_calc(load_path, save_path, save_name, method, kwargs, subset=None):
    function_mapping = {'ost': ost, 'mst': mst, 'ps': pk, 'resnet': resnet}
    raw_data = load_and_apply(load_path, function_mapping[method](**kwargs), device=torch.device('cuda:0'))

    if subset is not None:
        raw_data = raw_data[:subset]

    raw_data = raw_data.mean(axis=1)  # average over the patch axis

    if subset is None:
        subset = 'full'

    main_path = os.path.join(save_path, f'{save_name}_{subset}')

    # without altering data
    np.save(main_path + '.npy', raw_data)

    # with std
    scaler = Scaler(np.mean(raw_data), np.std(raw_data))
    std_data = scaler.transform(raw_data)
    np.save(main_path + '_std.npy', std_data)

    # with std and log
    log_data = np.log(raw_data)
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
    save_name = config['save_name']
    method = config['method']
    kwargs = config['kwargs']

    pre_calc(load_path, save_path, save_name, method, kwargs)

if __name__ == '__main__':
    main()