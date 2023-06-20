from ostlensing.dataloading import Scaler
import numpy as np
import torch
import os
import pandas as pd

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


def augment_patches():

    use_log = False

    op_dev = torch.device('cuda')
    end_dev = torch.device('cpu')

    load_path = '//pscratch/sd/m/mcraigie/cosmogrid/patches/'
    if use_log:
        save_path = '//pscratch/sd/m/mcraigie/cosmogrid/patches_log_std/'
    else:
        save_path = '//pscratch/sd/m/mcraigie/cosmogrid/patches_std/'

    all_dirs = os.listdir(os.path.join(load_path))
    all_dirs = np.sort(all_dirs)

    data = []
    for dir_ in all_dirs:
        fields = torch.from_numpy(np.load(os.path.join(load_path, dir_))).float()
        data.append(fields)

    data = torch.stack(data)

    if use_log:
        data = batch_apply(data, 16, torch.log, operate_device=op_dev, end_device=end_dev)

    #  calculate scaling values and apply
    scaler = Scaler(torch.mean(data), torch.std(data))
    result = batch_apply(data, 16, scaler.transform, operate_device=op_dev, end_device=end_dev)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the result as the dirs
    for i, dir_ in enumerate(all_dirs):
        data_half = result[i].cpu().numpy().astype(np.half)  # save as half precision for space
        np.save(os.path.join(save_path, dir_), data_half)


def augment_data():

    name = 'mst_precalc'
    load_path = f'//pscratch/sd/m/mcraigie/cosmogrid/{name}.npy'


    # without log
    save_path = f'//pscratch/sd/m/mcraigie/cosmogrid/{name}_std.npy'
    data = np.load(load_path)
    scaler = Scaler(np.mean(data), np.std(data))
    result = scaler.transform(data)
    np.save(save_path, result)

    # with log
    save_path = f'//pscratch/sd/m/mcraigie/cosmogrid/{name}_log_std.npy'
    data = np.load(load_path)
    data = np.log(data)
    scaler = Scaler(np.mean(data), np.std(data))
    result = scaler.transform(data)
    np.save(save_path, result)


def make_params():

    load_path = '//pscratch/sd/m/mcraigie/cosmogrid/params.csv'
    df = pd.read_csv(load_path)

    param_names = df.columns[1:14]

    print(param_names)
    param_values = df[param_names].values

    # cleaned df that drops irrelevant columns
    clean_df = pd.DataFrame(columns=param_names, data=param_values)

    # drop columns 'Ol' and 'm_nu' since they don't vary
    clean_df = clean_df.drop(columns=['Ol', 'm_nu'])

    # standardised df
    means = clean_df.mean()
    stds = clean_df.std()
    standardised_df = (clean_df - means) / stds

    # save the means and stds in their own df
    transform_df = pd.concat([means, stds], axis=1)
    transform_df = transform_df.transpose()
    transform_df.columns = clean_df.columns

    # add an index saying 'mean' and 'std'
    transform_df['index'] = ['mean', 'std']
    transform_df = transform_df.set_index('index')

    save_path = '//pscratch/sd/m/mcraigie/cosmogrid/params_clean.csv'
    clean_df.to_csv(save_path, index=False)

    save_path = '//pscratch/sd/m/mcraigie/cosmogrid/params_std.csv'
    standardised_df.to_csv(save_path, index=False)

    save_path = '//pscratch/sd/m/mcraigie/cosmogrid/params_transvals.csv'
    transform_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    make_params()



