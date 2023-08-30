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


def compute_power_spectrum(data, mask=None, deconv=False):

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
                      mask):

    print("processing {}".format(cosmo_dir))
    power_spectra = []
    permute_dirs = np.sort([pa for pa in os.listdir(os.path.join(main_path, cosmo_dir)) if 'perm' in pa])[:num_perms]

    for n, permute_dir in enumerate(permute_dirs):
        p = os.path.join(main_path, cosmo_dir, permute_dir, fname)
        f = h5py.File(p, 'r')
        full_map = f[map_type]['desy3metacal{}'.format(redshift_bin)][()]

        power_spectra.append(compute_power_spectrum(full_map, mask=mask))

    power_spectra = np.stack(power_spectra).reshape((num_perms * len(patch_centres), patch_size, patch_size))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(os.path.join(output_path, 'powspec_{}.npy'.format(cosmo_dir)), power_spectra)


def make_powspecs_cosmogrid(output_path,
                           num_perms=1,
                           map_type='kg',
                           redshift_bin=3):

    main_path = r'//global/cfs/cdirs/des/cosmogrid/DESY3/grid'
    fname = r'projected_probes_maps_baryonified512.h5'  # can also be nobaryons512.h5

    cosmo_dirs = os.listdir(main_path)
    cosmo_dirs = np.sort(cosmo_dirs)[:10]

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            mute = pickle.load(f)
            f.close()
        return mute

    # mask = load_obj('/global/cfs/cdirs/des//mass_maps/Maps_final//mask_DES_y3')
    # mask = hp.ud_grade(mask, nside_out=512)

    npix = hp.nside2npix(512)
    ipix = np.arange(npix)
    theta, phi = hp.pix2ang(512, ipix)
    mask = np.logical_and((theta <= np.pi / 2), (phi <= np.pi / 2))

    # run loop with mp
    pool = mp.Pool()

    process_function = partial(process_cosmo_dir,
                               main_path=main_path,
                               output_path=output_path,
                               num_perms=num_perms,
                               fname=fname,
                               map_type=map_type,
                               redshift_bin=redshift_bin,
                               mask=mask)

    pool.map(process_function, cosmo_dirs)
    pool.close()
    pool.join()

    cosmo_dirs = os.listdir(output_path)
    cosmo_dirs = np.sort(cosmo_dirs)

    final_path = r'//pscratch/sd/m/mcraigie/cosmogrid/precalc/cl_powspecs.npy'
    all_powspecs = []
    for cd in cosmo_dirs:
        cosmo_powspec = np.load(cd)
        all_powspecs.append(cosmo_powspec)

    result = np.stack(all_powspecs)  # shape num_cosmo, num_cls
    result = (result - np.mean(result, axis=0)) / np.std(result, axis=0)
    np.save(final_path, result)


def main():
    output_path = r'//pscratch/sd/m/mcraigie/cosmogrid/powspecs/unmasked/'
    make_powspecs_cosmogrid(output_path=output_path,
                           num_perms=1,
                           map_type='kg',
                           redshift_bin=2,
                           )


if __name__ == '__main__':
    main()

