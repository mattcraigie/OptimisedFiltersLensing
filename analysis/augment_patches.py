from ostlensing.dataloading import Scaler, batch_apply
import numpy as np
import torch
import os


def main():
    load_path = '//pscratch/sd/m/mcraigie/cosmogrid/patches/'
    save_path = '//pscratch/sd/m/mcraigie/cosmogrid/patches_log_std/'

    all_dirs = os.listdir(os.path.join(load_path))
    all_dirs = np.sort(all_dirs)

    data = []
    for dir_ in all_dirs:
        fields = torch.from_numpy(np.load(os.path.join(load_path, dir_))).float()
        data.append(fields)

    data = torch.stack(data)

    # first calculate log
    data_log = batch_apply(data, 8, torch.log, device=torch.device('cuda'))

    # then calculate scaling values and apply
    logged_mean = torch.mean(data_log)
    logged_std = torch.std(data_log)
    scaler = Scaler(logged_mean, logged_std)
    result = batch_apply(data, 8, scaler.transform, device=torch.device('cuda'))

    # save the result as the dirs
    for i, dir_ in enumerate(all_dirs):
        np.save(os.path.join(save_path, dir_), result[i].cpu().numpy())


if __name__ == '__main__':
    main()



