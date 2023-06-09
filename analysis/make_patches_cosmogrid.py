from ostlensing.dataloading import healpix_map_to_patches, compute_patch_centres
import os
import h5py
import multiprocessing as mp
from functools import partial
import numpy as np


def process_cosmo_dir(cosmo_dir,
                      main_path,
                      output_path,
                      num_perms,
                      fname,
                      map_type,
                      redshift_bin,
                      patch_centres,
                      patch_size,
                      resolution):
    cosmo_patches = []
    permute_dirs = np.sort([pa for pa in os.listdir(os.path.join(main_path, cosmo_dir)) if 'perm' in pa])[:num_perms]

    for n, permute_dir in enumerate(permute_dirs):
        p = os.path.join(main_path, cosmo_dir, permute_dir, fname)
        f = h5py.File(p, 'r')
        full_map = f[map_type]['desy3metacal{}'.format(redshift_bin)][()]
        cosmo_patches.append(healpix_map_to_patches(full_map, patch_centres, patch_size, resolution))

    cosmo_patches = np.stack(cosmo_patches).reshape((num_perms * len(patch_centres), patch_size, patch_size))
    np.save(os.path.join(output_path, 'patches_{}.npy'.format(cosmo_dir)), cosmo_patches)


def make_patches_cosmogrid(output_path,
                          patch_nside=4,
                          patch_size=128,
                          resolution=7,  # arcmin
                          threshold=0.2,
                          num_perms=1,
                          map_type='kg',
                          redshift_bin=3):

    main_path = r'//global/cfs/cdirs/des/cosmogrid/DESY3/grid'
    fname = r'projected_probes_maps_baryonified512.h5'  # can also be nobaryons512.h5

    cosmo_dirs = os.listdir(main_path)
    cosmo_dirs = np.sort(cosmo_dirs)

    # replace with DES mask? Or are they in-built?
    mask = None

    patch_centres = compute_patch_centres(patch_nside, mask, threshold)

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
                               resolution=resolution)

    pool.map(process_function, cosmo_dirs)
    pool.close()
    pool.join()


def main():
    output_path = "/pscratch/sd/m/mcraigie/cosmogrid/patches/"
    make_patches_cosmogrid(output_path=output_path,
                           patch_nside=4,
                           patch_size=128,
                           resolution=7,  # arcmin
                           threshold=0.2,
                           num_perms=1,
                           map_type='kg',
                           redshift_bin=3
                           )


if __name__ == '__main__':
    main()