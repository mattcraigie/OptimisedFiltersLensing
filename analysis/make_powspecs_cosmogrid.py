import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from ostlensing.dataloading import healpix_map_to_patches, compute_patch_centres
import os
import h5py
import multiprocessing as mp
from functools import partial
import numpy as np
import healpy as hp
import pickle


def compute_power_spectrum(data, nside, mask=None, deconv=False):

    masked_map = np.where(mask, data, hp.UNSEEN)
    alm = hp.map2alm(masked_map)
    power_spectrum = hp.alm2cl(alm)

    if deconv:
        mask_power_spectrum = hp.anafast(mask)
        return power_spectrum / mask_power_spectrum

    return power_spectrum


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

        if mask is not None:

            # map normalisation -- I'm concerned this isn't the best way to do it. I should think about this.
            full_map[mask] = np.log(full_map[mask])
            full_map[mask] = (full_map[mask] - np.mean(full_map[mask])) / np.std(full_map[mask])
            full_map[~mask] = 0
        else:
            full_map = np.log(full_map)
            full_map = (full_map - np.mean(full_map)) / np.std(full_map)

        cosmo_patches.append(healpix_map_to_patches(full_map, patch_centres, patch_size, resolution))

    cosmo_patches = np.stack(cosmo_patches).reshape((num_perms * len(patch_centres), patch_size, patch_size))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(os.path.join(output_path, 'patches_{}.npy'.format(cosmo_dir)), cosmo_patches)


def make_powspecs_cosmogrid(output_path,
                           num_perms=1,
                           map_type='kg',
                           redshift_bin=3,
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

    npix = hp.nside2npix(512)
    ipix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, ipix)
    mask = (theta <= np.pi / 2)  # Octant mask



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
                           redshift_bin=2,
                           subset=30,
                           )


if __name__ == '__main__':
    main()

