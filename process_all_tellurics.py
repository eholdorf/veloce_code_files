import rv
import astropy.io.fits as pyfits
import numpy as np

root_dir = '/priv/avatar/velocedata/anu_processing/'
data_dir = '/priv/avatar/velocedata/Data/spec_211202/'

#--------
allfits = pyfits.getdata(root_dir + 'veloce_observations.fits')
bstars = np.where(allfits['obs_type']=='BSTAR')[0]
for star_ix in bstars:
    print(allfits['star_names'][star_ix])
    for file_ix in range(allfits['number_obs'][star_ix]):
        fitsname = allfits['fits_names'][star_ix, file_ix]
        dot = fitsname.find('.')
        fitsname = fitsname[:dot] + 'i_extf.fits'
        full_filename = data_dir + allfits['directory'][star_ix, file_ix] + '/' + fitsname
        try:
            hh=pyfits.getheader(full_filename)
            print(full_filename)
        except:
            print("Error: Can't find " + full_filename)

