import torch
import argparse
import yaml

from scattering_transform.scattering_transform import ScatteringTransform2d, Reducer
from scattering_transform.filters import Morlet, FixedFilterBank
from scattering_transform.power_spectrum import PowerSpectrum

from ostlensing.dataloading import load_and_apply


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


def pk(size, num_bins, device):
    ps = PowerSpectrum(size, num_bins)
    ps.to(device)
    return ps


def pre_calc(load_path, save_path, method, kwargs):
    function_mapping = {'ost': ost, 'mst': mst, 'ps': pk}
    load_and_apply(load_path, function_mapping[method](**kwargs), device=torch.device('cuda:0'), save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/ps.yml', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    load_path = config['load_path']
    save_path = config['save_path']
    method = config['method']
    kwargs = config['kwargs']

    pre_calc(load_path, save_path, method, kwargs)

if __name__ == '__main__':
    main()