import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nside = 512  # Adjust as needed

# Create an octant mask
npix = hp.nside2npix(nside)
ipix = np.arange(npix)
theta, phi = hp.pix2ang(nside, ipix)
mask = (theta <= np.pi / 2)  # Octant mask

# Generate a HEALPix map for demonstration
map = np.random.normal(0, 1, npix)

# Apply the octant mask to the map
masked_map = np.where(mask, map, hp.UNSEEN)

# Compute spherical harmonic coefficients
alm = hp.map2alm(masked_map)

# Compute mask power spectrum
mask_power_spectrum = hp.anafast(mask)

# Compute inverse mask power spectrum
inverse_mask_power_spectrum = 1.0 / mask_power_spectrum

# Compute power spectrum of unmasked section
unmasked_power_spectrum = hp.alm2cl(alm)

# Deconvolve to remove mask effects
deconvolved_power_spectrum = unmasked_power_spectrum * inverse_mask_power_spectrum

# Plot the original and deconvolved power spectra
plt.plot(unmasked_power_spectrum, label='Original')
plt.plot(deconvolved_power_spectrum, label='Deconvolved')
plt.xlabel('Multipole l')
plt.ylabel('Power Spectrum')
plt.title('Original vs. Deconvolved Power Spectrum')
plt.legend()
plt.show()