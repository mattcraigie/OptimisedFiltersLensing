from ostlensing.dataloading import Scaler
import numpy as np
import torch
import os

def batch_apply(data, bs, func, operate_device, end_device=None):
    if end_device is None:
        end_device = operate_device
    results = []
    num_batches = data.shape[0] // bs
    num_batches = num_batches if data.shape[0] % bs == 0 else num_batches + 1
    for i in range(num_batches):
        x = data[bs*i:bs*(i+1)]
        x = x.to(operate_device)
        results.append(func(x).to(end_device))
    return torch.cat(results, dim=0)


def main():

    op_dev = torch.device('cuda')
    end_dev = torch.device('cpu')

    load_path = '//pscratch/sd/m/mcraigie/cosmogrid/patches/'
    save_path = '//pscratch/sd/m/mcraigie/cosmogrid/patches_log_std/'

    all_dirs = os.listdir(os.path.join(load_path))
    all_dirs = np.sort(all_dirs)

    data = []
    for dir_ in all_dirs:
        fields = torch.from_numpy(np.load(os.path.join(load_path, dir_))).float()
        data.append(fields)

    data = torch.stack(data)
    print(data.shape)

    # first calculate log
    data_log = batch_apply(data, 1, torch.log, operate_device=op_dev, end_device=end_dev)


    # then calculate scaling values and apply
    logged_mean = torch.mean(data_log)
    logged_std = torch.std(data_log)
    scaler = Scaler(logged_mean, logged_std)
    result = batch_apply(data, 1, scaler.transform, operate_device=op_dev, end_device=end_dev)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the result as the dirs
    for i, dir_ in enumerate(all_dirs):
        np.save(os.path.join(save_path, dir_), result[i].cpu().numpy())


if __name__ == '__main__':
    main()



