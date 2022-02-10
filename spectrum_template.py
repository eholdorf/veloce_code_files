import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

file_path = '/priv/avatar/velocedata/Data/spec_211202/'

# Tau Ceti (HD10700) template
# this assumes that each of the observations have the same wavelength scale (i.e. points)
#now need to looking into the fact that they vary very slightly (so if the same, then average, if not then put into order to give a really high grid of points)
TC_observation_dir = ['191211/11dec30096oi_extf.fits', '191211/11dec30097oi_extf.fits', '191212/12dec30132oi_extf.fits', '191212/12dec30133oi_extf.fits','191212/12dec30134oi_extf.fits', '191213/13dec30076oi_extf.fits', '191213/13dec30077oi_extf.fits', '191214/14dec30066oi_extf.fits', '191214/14dec30067oi_extf.fits', '191214/14dec30068oi_extf.fits', '191215/15dec30097oi_extf.fits', '191215/15dec30098oi_extf.fits', '191215/15dec30099oi_extf.fits']

TC_flux = np.zeros((3900,40,))
TC_wavelength = np.zeros((3900,40,))
first_obs = pyfits.open(file_path+TC_observation_dir[0])
# done for the 5th fibre, which is the first stellar fibre
for path in TC_observation_dir:
	dd = pyfits.open(file_path+path)
	TC_flux += np.array(dd[0].data[:,:,5])
	TC_wavelength += np.array(dd[1].data[:,:,5])
	wavelength_diff = abs(first_obs[1].data[:,:,5] - dd[1].data[:,:,5])
	plt.plot(first_obs[1].data[:,:,5],100*wavelength_diff/first_obs[1].data[:,:,5],'.')
	plt.title('Wavelength Difference')
	plt.xlabel('Wavelength')
	plt.ylabel('Percentage Difference')
plt.show()


TC_wavelength = TC_wavelength/len(TC_observation_dir)
TC_flux = TC_flux/len(TC_observation_dir)
plt.figure()
plt.plot(TC_wavelength,TC_flux,'k',label = 'Data')
plt.title('Tau Ceti (11dec19 - 15dec19)')
plt.ylabel('Flux')
plt.xlabel('$\lambda$ ($\AA$)')
plt.show()

		

