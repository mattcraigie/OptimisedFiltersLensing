import torch
import argparse
import yaml
import os
import numpy as np

from scattering_transform.scattering_transform import ScatteringTransform2d, Reducer
from scattering_transform.filters import Morlet, FixedFilterBank, Box, BandPass, LowPass, ClippedMorlet
from scattering_transform.power_spectrum import PowerSpectrum

from ostlensing.dataloading import load_and_apply, Scaler
from ostlensing.models import ResNetWrapper


def st_func(filters, reduction, device, clip_sizes=None):
    st = ScatteringTransform2d(filters, clip_sizes=clip_sizes)
    st.to(device)
    reducer = Reducer(filters, reduction=reduction, normalise_s2=True)
    return lambda x: reducer(st(x))


def mst(size, num_scales, num_angles, reduction, device):
    morlet = Morlet(size, num_scales, num_angles)
    return st_func(morlet, reduction, device)


def clipped_mst(size, num_scales, num_angles, reduction, device):
    morlet = ClippedMorlet(size, num_scales, num_angles)
    return st_func(morlet, reduction, device, clip_sizes=[size // 2 ** j for j in range(num_scales)])


def ost(filter_path, reduction, device):
    filters = torch.load(filter_path)
    num_scales, num_angles, size, _ = filters.shape
    filter_bank = FixedFilterBank(filters)
    return st_func(filter_bank, reduction, device, clip_sizes=[size // 2 ** j for j in range(num_scales)])


def resnet(model_state_dict_path, pretrained_model, device):
    model = ResNetWrapper(pretrained_model=pretrained_model)
    if model_state_dict_path is not None:
        model.load_state_dict(torch.load(model_state_dict_path))
    model.to(device)
    return lambda x: model(x.unsqueeze(0)).squeeze(0)


def pk(size, num_bins, device):
    ps = PowerSpectrum(size, num_bins)
    ps.to(device)
    return ps


def box(size, num_scales, num_angles, reduction, device):
    b = Box(size, num_scales, num_angles)
    return st_func(b, reduction, device, clip_sizes=[size // 2 ** j for j in range(num_scales)])


def bandpass(size, num_scales, num_angles, reduction, device):
    b = BandPass(size, num_scales, num_angles)
    return st_func(b, reduction, device, clip_sizes=[size // 2 ** j for j in range(num_scales)])


def lowpass(size, num_scales, num_angles, reduction, device):
    b = LowPass(size, num_scales, num_angles)
    return st_func(b, reduction, device, clip_sizes=[size // 2 ** j for j in range(num_scales)])


def pre_calc(load_path, save_path, file_name, method, kwargs):
    function_mapping = {'ost': ost,
                        'mst': mst,
                        'ps': pk,
                        'resnet': resnet,
                        'box': box,
                        'bandpass': bandpass,
                        'lowpass': lowpass}

    with torch.no_grad():
        data = load_and_apply(load_path, function_mapping[method](**kwargs), device=torch.device('cuda:0'))

    main_path = os.path.join(save_path, file_name)

    # without altering data
    np.save(main_path + '.npy', data)

    # with std
    scaler = Scaler(np.mean(data, axis=0), np.std(data, axis=0))
    std_data = scaler.transform(data)
    np.save(main_path + '_std.npy', std_data)

    # with std and log
    log_data = np.log(data)
    scaler = Scaler(np.mean(log_data, axis=0), np.std(log_data, axis=0))
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

    pre_calc(load_path, save_path, file_name, method, kwargs)



if __name__ == '__main__':
    main()