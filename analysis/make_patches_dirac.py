from ostlensing.dataloading import healpix_map_to_patches, compute_patch_centres
import os
import h5py
import multiprocessing as mp
from functools import partial
import numpy as np
import healpy as hp
import pickle


def process_cosmo_file(fname,
                       main_path,
                       output_path,
                       patch_centres,
                       patch_size,
                       resolution,
                       mask):
    print("processing {}".format(fname))

    path = os.path.join(main_path, fname)
    full_map = hp.fitsfunc.read_map(path)

    # map normalisation
    # full_map[mask] = np.log(full_map[mask])
    full_map[mask] = full_map[mask] / np.std(full_map[mask])
    full_map[~mask] = 0

    cosmo_patches = healpix_map_to_patches(full_map, patch_centres, patch_size, resolution)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    name = fname[15:23]
    np.save(os.path.join(output_path, 'patches_{}.npy'.format(name)), cosmo_patches)


def make_patches_cosmogrid(output_path,
                          patch_nside=4,
                          patch_size=128,
                          resolution=7,  # arcmin
                          threshold=0.2,
                          redshift_bin=3):

    main_path = r'//global/cfs/cdirs/des/mgatti/Dirac_mocks/kE_maps'

    cosmo_dirs = os.listdir(main_path)
    cosmo_dirs = np.sort(cosmo_dirs)

    includes = []

    for cd in cosmo_dirs:
        if 'zbin'+str(redshift_bin) in cd:
            includes.append(cd)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            mute = pickle.load(f)
            f.close()
        return mute

    mask = load_obj('/global/cfs/cdirs/des//mass_maps/Maps_final//mask_DES_y3')
    mask = hp.ud_grade(mask, nside_out=512)
    patch_centres = compute_patch_centres(patch_nside, mask.copy().astype(np.float64), threshold)

    # run loop with mp
    pool = mp.Pool()

    process_function = partial(process_cosmo_file,
                               main_path=main_path,
                               output_path=output_path,
                               patch_centres=patch_centres,
                               patch_size=patch_size,
                               resolution=resolution,
                               mask=mask)

    pool.map(process_function, cosmo_dirs)
    pool.close()
    pool.join()


def main():
    output_path = "/pscratch/sd/m/mcraigie/dirac/patches/masked/"
    make_patches_cosmogrid(output_path=output_path,
                           patch_nside=4,
                           patch_size=128,
                           resolution=7,  # arcmin
                           threshold=0.2,
                           redshift_bin=3
                           )


if __name__ == '__main__':
    main()
