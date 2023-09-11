from ostlensing.dataloading import healpix_map_to_patches, compute_patch_centres
import os
import h5py
import multiprocessing as mp
from functools import partial
import numpy as np
import healpy as hp
import pickle


def process_cosmo_dir(cosmo_dir,
                      main_path,
                      output_path,
                      num_perms,
                      fname,
                      map_type,
                      redshift_bin,
                      patch_centres,
                      patch_size,
                      resolution,
                      mask):

    print("processing {}".format(cosmo_dir))
    cosmo_patches = []
    permute_dirs = np.sort([pa for pa in os.listdir(os.path.join(main_path, cosmo_dir)) if 'perm' in pa])[:num_perms]

    for n, permute_dir in enumerate(permute_dirs):
        p = os.path.join(main_path, cosmo_dir, permute_dir, fname)
        f = h5py.File(p, 'r')
        full_map = f[map_type]['desy3metacal{}'.format(redshift_bin)][()]

        full_map = np.log(full_map)
        if map_type == 'kg':
            # normalise to approximately mean 0, std 1 for the log convergence fields
            full_map = (full_map + 4.5) / 0.05
        elif map_type == 'dg':
            # normalise to approximately mean 0, std 1 for the log density fields
            full_map = (full_map + 0.03) / 0.07

        if mask is not None:
            full_map[~mask] = 0

        cosmo_patches.append(healpix_map_to_patches(full_map, patch_centres, patch_size, resolution))

    cosmo_patches = np.stack(cosmo_patches).reshape((num_perms * len(patch_centres), patch_size, patch_size))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(os.path.join(output_path, 'patches_{}.npy'.format(cosmo_dir)), cosmo_patches)


def make_patches_cosmogrid(output_path,
                           patch_nside=4,
                           patch_size=128,
                           resolution=7,  # arcmin
                           threshold=0.2,
                           num_perms=1,
                           map_type='kg',
                           redshift_bin=1,
                           subset=None):

    main_path = r'//global/cfs/cdirs/des/cosmogrid/DESY3/grid'
    fname = r'projected_probes_maps_baryonified512.h5'  # can also be nobaryons512.h5

    cosmo_dirs = os.listdir(main_path)
    cosmo_dirs = np.sort(cosmo_dirs)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            mute = pickle.load(f)
            f.close()
        return mute

    # mask = load_obj('/global/cfs/cdirs/des//mass_maps/Maps_final//mask_DES_y3')
    # mask = hp.ud_grade(mask, nside_out=512)
    # patch_centres = compute_patch_centres(patch_nside, mask.copy().astype(np.float64), threshold)

    mask = None
    patch_centres = compute_patch_centres(patch_nside, mask, threshold)

    if subset is not None:
        patch_centres = patch_centres[:subset]

    # run loop with mp
    pool = mp.Pool()

    process_function = partial(process_cosmo_dir,
                               main_path=main_path,
                               output_path=output_path,
                               num_perms=num_perms,
                               fname=fname,
                               map_type=map_type,
                               redshift_bin=redshift_bin,
                               patch_centres=patch_centres,
                               patch_size=patch_size,
                               resolution=resolution,
                               mask=mask)

    pool.map(process_function, cosmo_dirs)
    pool.close()
    pool.join()


def main():
    output_path = "/pscratch/sd/m/mcraigie/cosmogrid/patches/clustering/unmasked/"
    make_patches_cosmogrid(output_path=output_path,
                           patch_nside=4,
                           patch_size=128,
                           resolution=7,  # arcmin
                           threshold=0.2,
                           num_perms=1,
                           map_type='kg',
                           redshift_bin=1,
                           subset=None,
                           )


if __name__ == '__main__':
    main()