import numpy as np
import astropy.io.fits as pyfits
import glob
import matplotlib.pyplot as plt

ddir = '/home/ehold13/veloce_scripts/veloce_reduction/10700/'
allfiles = glob.glob(ddir + '*fits')

#First, initialise arrays with first file
ff = pyfits.open(allfiles[0])
rvs = ff[1].data
rv_errs = ff[2].data
medflux = ff[3].data
mjds = ff[4].data['MJDs']
bcs = ff[4].data['BCs']
orig_files = ff[4].data['Files']

for fn in allfiles[1:]:
    ff = pyfits.open(fn)
    rvs = np.concatenate( (rvs, ff[1].data) )
    rv_errs = np.concatenate( (rv_errs, ff[2].data) )
    medflux = np.concatenate( (medflux, ff[3].data) )
    mjds = np.concatenate( (mjds, ff[4].data['MJDs']) )
    bcs = np.concatenate( (bcs, ff[4].data['BCs']) )
    orig_files = np.concatenate( (orig_files, ff[4].data['Files']) )

#!!!This should be a weighted mean
#!!!Also, cutting on mean RV isn't good. 
good_orders_mn = np.mean(rvs[:,30:36,:], axis=1)
good_orders_err = np.mean(rv_errs[:,30:36,:], axis=1)

#Look for fiber-to-fiber outliers
fiber_std = np.std(good_orders_mn, axis=1)
straight_mean_rv = np.mean(good_orders_mn, axis=1)
weighted_mean_rv = np.sum(good_orders_mn/good_orders_err**2, axis=1)/np.sum(1/good_orders_err**2, axis=1)

good_obs = (fiber_std < 3*np.median(fiber_std)) & (np.abs(weighted_mean_rv) < 0.08)

#plt.imshow(good_orders_mn[good_obs], aspect='auto', interpolation='nearest')

fiber_rvs = good_orders_mn
for ix,obs_rv  in enumerate(fiber_rvs):
    obs_rv -= weighted_mean_rv[ix]

fiber_means = np.mean(fiber_rvs[good_obs], axis=0)
fiber_std = np.std(fiber_rvs[good_obs], axis=0)

#Lets correct for this and see if it helps.By only subtracting fiber-to-fiber means, this should 
#not bias the radial velocity observations. To check!s
corrected_gd_mn = good_orders_mn
corrected_gd_mn -= fiber_means
corrected_wt_mn = np.sum(corrected_gd_mn/good_orders_err**2, axis=1)/np.sum(1/good_orders_err**2, axis=1)

plt.plot(mjds[good_obs], corrected_wt_mn[good_obs],'.')
