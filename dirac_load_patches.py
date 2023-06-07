"""
This code is used to load the patches from the Dirac simulations and return them as data.

PatchMaker class adapted from Marco Gatti's Moments Analysis package (moments_map class).
Available at https://github.com/mgatti29/Moments_analysis/blob/main/Moments_analysis/Compute_moments.py

"""

import numpy as np
import os
import copy
import gc
import glob
import pickle
import healpy as hp


def index2radec(index, nside, nest=False):
    """
    Converts HEALPix index to Declination and Right Ascension.

    Args:
        index (array): HEALPix pixel indices.
        nside (int): HEALPix nside parameter.
        nest (bool, optional): Nesting scheme of the HEALPix pixels. Defaults to False.

    Returns:
        tuple: Declination and Right Ascension.
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)
    return -np.degrees(theta - np.pi / 2.), np.degrees(phi)


def radec2pix(ra, dec, nside):
    """
    Converts RA, DEC to HEALPix pixel coordinates.

    Args:
        ra (array): Right Ascension values.
        dec (array): Declination values.
        nside (int, optional): HEALPix nside parameter

    Returns:
        array: HEALPix pixel coordinates.
    """
    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    return pix


class PatchMaker(object):
    def __init__(self, conf={'output_folder': './'}):
        '''
        Initialise the moments_map object.
        
        Parameters:
        - conf (dict): Configuration parameters for the moments_map object.
                       Default is {'output_folder': './'}.
        '''
        self.conf = conf
        self.fields = dict()
        
        # Create the output folder if it doesn't exist
        try:
            if not os.path.exists(self.conf['output_folder']):
                os.mkdir((self.conf['output_folder']))
        except:
            pass
        
        # Create the smoothed maps folder if it doesn't exist
        try:
            if not os.path.exists((self.conf['output_folder'])+'/smoothed_maps/'):
                os.mkdir((self.conf['output_folder'])+'/smoothed_maps/')
        except:
            pass
    
    def add_map(self, map_, field_label='', tomo_bin=0):
        '''
        Add a map to the fields dictionary.
        
        Parameters:
        - map_ (array-like): The map to be added.
        - field_label (str): The label for the field.
        - tomo_bin (int): The tomo_bin value.
        '''
        if field_label in self.fields.keys():
            self.fields[field_label][tomo_bin] = copy.deepcopy(map_)
        else:
            self.fields[field_label] = dict()
            self.fields[field_label][tomo_bin] = copy.deepcopy(map_)


    def cut_patches(self, nside=512, nside_small=16, threshold=0.0001):
        """
        It takes every entry in the field catalog, and cuts them into squared patches using the healpy gnomview projection.
        delta is the side length of the square patch in degrees.
        patches are saved into the 'field_patches' dictionary.
        """
        
        # initial guess for the size of the patch
        delta = np.sqrt(hp.nside2pixarea(nside_small, degrees=True))*4
        
        # test maps -----------------------------
        mask_small = hp.ud_grade(self.mask, nside_out=nside_small)
        map_test_uniform = np.ones(len(self.mask))
        mask_pix = mask_small!=0.


        # this is to know which part to mask ---
        map_indexes = np.arange(hp.nside2npix(nside_small))
        map_indexes_large = np.arange(hp.nside2npix(nside))
        dec_large, ra_large = index2radec(map_indexes_large, nside, nest=False)
        pix_large_convert = radec2pix(ra_large, dec_large, nside=nside_small)

        # these are the centers
        dec_, ra_ = index2radec(map_indexes, nside_small, nest=False)
        pairs_ = np.vstack([ra_[mask_pix], dec_[mask_pix], map_indexes[mask_pix]])

        pairs = []

        count_area = 0

        print('number of patches: ', len(map_indexes))
        pixels = np.int(delta/(hp.nside2resol(nside, arcmin=True)/60))
        res = hp.nside2resol(nside, arcmin=True)
        xsize = 2**np.int(np.log2(pixels))

        print('resolution [arcmin]: ', res)
        print('pixels per side: ', xsize)
        print('frame size [deg]: ', xsize*res/60.)

        # checks which patch to keep

        for i in range(pairs_.shape[1]):
            ra_, dec_, i_ = pairs_[:, i]


            map_test_ = copy.deepcopy(self.mask)
            map_test_uniform_ = copy.deepcopy(map_test_uniform)
            mask_ = np.in1d(pix_large_convert, i_)
            map_test_[~mask_] = 0.
            map_test_uniform_[~mask_] = 0.
            m_ref = hp.gnomview(map_test_, rot=(ra_, dec_), xsize=xsize, no_plot=True, reso=res,
                                return_projected_map=True)

            if (1.*np.sum((m_ref.data).flatten()) / np.sum(map_test_uniform_.flatten())) > threshold:
                # more than 20% of the stamp
                pairs.append([ra_, dec_, i_])
                count_area += (res/60.)**2 *np.sum(m_ref)

        print('TOTAL AREA: ', count_area)

        self.fields_patches = dict()
        
        for key in self.fields.keys():
            self.fields_patches[key] = dict()
            for key2 in self.fields[key].keys():

                patches = []

                for i in range(len(pairs)):
                    ra_, dec_, i_ = pairs[i]
                    
                    fieldc = copy.deepcopy(self.fields[key][key2])
                    mask_ = np.in1d(pix_large_convert, i_)
                    fieldc[~mask_] = 0.
                    mt1 = hp.gnomview(fieldc, rot=(ra_, dec_), xsize=xsize, no_plot=True, reso=res,
                                      return_projected_map=True)

                    patches.append(mt1.data)
                    
 
                np.save(self.conf['output_folder']+'{0}_{1}'.format(key, key2), patches)
                self.patch_size = patches[0].shape
                self.fields_patches[key][key2] = self.conf['output_folder']+'{0}_{1}'.format(key, key2)


def save_obj(name, obj):
    """
    Saves an object to a pickle file.

    Args:
        name (str): The name of the pickle file.
        obj (object): The object to be saved.
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)
        f.close()


def load_obj(name):
    """
    Loads an object from a pickle file.

    Args:
        name (str): The name of the pickle file.

    Returns:
        object: The loaded object.
    """
    with open(name + '.pkl', 'rb') as f:
        mute = pickle.load(f)
        f.close()
    return mute


def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048, nosh=True):
    """
    Converts shear to convergence on a sphere using HEALPix maps.

    Args:
        gamma1 (array): Gamma1 shear map.
        gamma2 (array): Gamma2 shear map.
        mask (array): Mask for the maps.
        nside (int, optional): HEALPix nside parameter. Defaults to 1024.
        lmax (int, optional): Maximum multipole moment. Defaults to 2048.
        nosh (bool, optional): Flag indicating whether to remove the monopole and dipole terms. Defaults to True.

    Returns:
        tuple: E-mode convergence map, B-mode convergence map, and E-mode alms.
    """
    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)
    if nosh:
        almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
        almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    else:
        almsE = alms[1] * 1.
        almsB = alms[2] * 1.
    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0

    almssm = [alms[0], almsE, almsB]

    kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
    B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    return E_map, B_map, almsE


def make_dirac_patches(file, output=''):
    dict_temp = np.load(file, allow_pickle=True).item()
    target = file.split('/pscratch/sd/m/mgatti/Dirac/')[1].split('.npy')[0]
    mock = target.split('runs')[1].split('_')[0]
    run = int(target.split('run')[2].split('_')[0])
    f = open(('/global/u2/m/mgatti/Mass_Mapping/peaks/params_run_1_Niall_{0}.txt'.format(mock)), 'r')
    om_ = []
    h_ = []
    ns_ = []
    ob_ = []
    s8_ = []
    w_ = []
    for i, f_ in enumerate(f):
        if i > 0:
            om_.append(float(f_.split(',')[0]))
            h_.append(float(f_.split(',')[4]))
            ns_.append(float(f_.split(',')[5]))
            ob_.append(float(f_.split(',')[3]))
            s8_.append(float(f_.split(',')[1]))
            w_.append(float(f_.split(',')[2]))
        else:
            print(f_)

    # iterate over the different realisations for the same cosmology
    for rel in range(4):
        params = dict()
        params['om'] = om_[run - 1]
        params['h'] = h_[run - 1]
        params['s8'] = s8_[run - 1]
        params['w'] = w_[run - 1]
        params['ob'] = ob_[run - 1]
        params['ns'] = ns_[run - 1]
        for k in dict_temp[rel]['nuisance_params'].keys():
            params[k] = dict_temp[rel]['nuisance_params'][k]

        if not os.path.exists(output + target + '_rel{0}'.format(rel) + '.pkl'):

            conf = dict()
            conf['nside'] = 512
            conf['output_folder'] = output_intermediate + '/test_' + target + '_rel{0}'.format(rel)

            patchmaker = PatchMaker(conf)
            # mask_DES_y3 = load_obj('/global/cfs/cdirs/des//mass_maps/Maps_final//mask_DES_y3')

            # iterate over the different tomographic bins
            # In my case, I think it makes sense to only use one tomographic bin, given we are trying to optimise the
            # filters, which are sensitive to the scales, and scales vary across tomographic bins.
            # The question becomes which bin is best? This will be a balance between signal (from high angular count)
            # and non-Gaussian structure (which is larger for the closer bins)


            tomo_bins = [0,]  # 1, 2, 3]

            for t in tomo_bins:
                e1 = np.zeros(hp.nside2npix(conf['nside']))
                e2 = np.zeros(hp.nside2npix(conf['nside']))
                e1[dict_temp[rel][t + 1]['pix']] = dict_temp[rel][t + 1]['e1']
                e2[dict_temp[rel][t + 1]['pix']] = dict_temp[rel][t + 1]['e2']
                mask_sims = np.in1d(np.arange(len(e1)), dict_temp[rel][t + 1]['pix'])
                f, fb, almsE, almsB = g2k_sphere(e1, e2, mask_sims, nside=conf['nside'], lmax=conf['nside'] * 2,
                                                 nosh=True)

                patchmaker.add_map(f, field_label='k', tomo_bin=t)
                patchmaker.add_map(fb, field_label='bk', tomo_bin=t)

                # I also don't think I need the noise maps
                # e1n = np.zeros(hp.nside2npix(conf['nside']))
                # e2n = np.zeros(hp.nside2npix(conf['nside']))
                # e1n[dict_temp[rel][t + 1]['pix']] = dict_temp[rel][t + 1]['e1n']
                # e2n[dict_temp[rel][t + 1]['pix']] = dict_temp[rel][t + 1]['e2n']
                # fn, fbn, almsEN, almsBN = g2k_sphere(e1n, e2n, mask_sims, nside=conf['nside'], lmax=conf['nside'] * 2,
                #                                      nosh=True)
                # patchmaker.add_map(fn, field_label='kn', tomo_bin=t)
                # patchmaker.add_map(fbn, field_label='bkn', tomo_bin=t)

                if t == 0:
                    # Only add the mask on the first iteration
                    patchmaker.mask = mask_sims
                    # patchmaker.mask = dict_temp[rel][t+1]['mask']

            patchmaker.cut_patches(nside=512, nside_small=8)

#
# import shutil
# shutil.rmtree(output_intermediate+'/test_'+target+'_rel{0}'.format(rel))
# save_obj(output+'moments_'+target+'_rel{0}'.format(rel),[mcal_moments,params])

# srun --nodes=1 --tasks-per-node=128   python run_PWH.py


if __name__ == '__main__':

    output_intermediate = '/pscratch/sd/m/mgatti/PWHM/temp/'
    output = '//global/cfs/cdirs/des/mgatti/Dirac/output_moments_ST_new_dirac_C/'
    files = glob.glob('/pscratch/sd/m/mgatti/Dirac/*')
    f_ = []
    for f in files:
        try:
            xx = np.int(f.split('noiserel')[1].split('.npy')[0])

            if ('512' in f) and (xx > 4):
                f_.append(f)
        except:
            pass

    runstodo = []
    count = 0
    for f in f_:
        target = f.split('/pscratch/sd/m/mgatti/Dirac/')[1].split('.npy')[0]
        if not os.path.exists(output + target + '_rel3.pkl'):
            runstodo.append(f)
        else:
            count += 1

