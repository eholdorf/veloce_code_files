import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.table import Table
from .main_funcs import log_scale_interpolation
from .main_funcs import telluric_correction
from .main_funcs import find_telluric_star
from . import get_observations
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as optimise
import astropy.constants as c
import astropy.units as u
from barycorrpy import get_BC_vel
from . import utils
from rv.main_funcs import barycentric_correction
import os
from astropy.time import Time
import time
from progressbar import progressbar
from progress.bar import ShadyBar
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame
import glob
import radvel
import radvel.likelihood
from radvel.plot import orbit_plots, mcmc_plots
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 18}
plt.rc('font', **font)


def create_observation_fits(standard, obs_fits, date, save_dir, combine_fibres = False):
    """
    Description
    -----------
    This function creates the corrected .fits files which can be used to derive radial velocities.
    
    Parameters
    ----------
    standard : type - string
        Name of the fits file want final fits file to have wavelength scale of, e.g. '11dec30096o.fits'
    
    obs_fits : type - string
        Name of the fits file to correct, e.g. '11dec30096o.fits'
    
    date : type - string
        Date of the observation in yymmdd format  
    
    save_dir : type - string
        The directroy to save the resulting .fits file.
    
    combine_fibres : type - boolean (default - False)
        If want to combine the fibre fluxes then True, else False
    
    Returns
    -------
    spect : type - numpy nd-array
        Spectrum of the corrected .fits file
    
    wavelength : type - numpy nd-array
        Wavelength for the corrected .fits file
    
    spect_err : type - numpy nd-array
        Spectrum error of the corrected .fits file
    """
    # find the file path
    file_path = get_observations.get_fits_path([obs_fits.encode('utf-8')])
    for path in sum(file_path,[]):
        if path[41:47] == date:
            fits_path = path
    
    # read in the data and select the stellar fibres   
    dd = pyfits.open(fits_path)
    all_log_w = dd[1].data[:,:,4:23]
    all_s_logflux = dd[0].data[:,:,4:23]
    all_s_logerrflux = dd[2].data[:,:,4:23]
    airmass_star = dd[0].header['AIRMASS']
    
    
    target_info_a, telluric_info_a = find_telluric_star(obs_fits, 'after', date)
    target_info_b, telluric_info_b = find_telluric_star(obs_fits, 'before', date)
    
    # find the telluric spectrum for before and after the observation
    B_plus_saved = None
    if not os.path.exists('/priv/avatar/ehold13/tellurics/'+str(telluric_info_b[2].decode('utf-8'))+'_'+str(telluric_info_b[1].decode('utf-8'))[0:10]+'.fits'):
        wave_tell_b, telluric_spec_b, telluric_err_spec_b, target_info_b, telluric_info_b, B_plus_saved = telluric_correction(obs_fits,'before', fits_path[41:47] ,scrunch = True, B_plus = B_plus_saved, airmass_corr = False)
    if not os.path.exists('/priv/avatar/ehold13/tellurics/'+str(telluric_info_a[2].decode('utf-8'))+'_'+str(telluric_info_a[1].decode('utf-8'))[0:10]+'.fits'):
        wave_tell_a, telluric_spec_a, telluric_err_spec_a, target_info_a, telluric_info_a, B_plus_saved = telluric_correction(obs_fits,'after', fits_path[41:47], scrunch = True, B_plus = B_plus_saved,airmass_corr = False)
   
    
    telluric_a = pyfits.open('/priv/avatar/ehold13/tellurics/'+str(telluric_info_a[2].decode('utf-8'))+'_'+str(telluric_info_a[1].decode('utf-8'))[0:10]+'.fits')
    telluric_a_airmass = float(telluric_a[3].data['AIRMASS'][0])
    
    telluric_spec_a = telluric_a[0].data
    telluric_err_spec_a = telluric_a[2].data*(airmass_star/telluric_a_airmass) /telluric_spec_a 
    telluric_spec_a = telluric_a[0].data ** (airmass_star/telluric_a_airmass)
    telluric_err_spec_a *= telluric_spec_a
    
    wave_tell_a = telluric_a[1].data
    
    telluric_b = pyfits.open('/priv/avatar/ehold13/tellurics/'+str(telluric_info_b[2].decode('utf-8'))+'_'+str(telluric_info_b[1].decode('utf-8'))[0:10]+'.fits')
    
    telluric_b_airmass = float(telluric_b[3].data['AIRMASS'][0])
    
    telluric_spec_b = telluric_b[0].data
    telluric_err_spec_b = telluric_b[2].data*(airmass_star/telluric_b_airmass) /telluric_spec_b
    telluric_spec_b = telluric_b[0].data ** (airmass_star/telluric_b_airmass)
    telluric_err_spec_b *= telluric_spec_b
    
    wave_tell_b = telluric_b[1].data
    
    # do a time weighted average of the before and after tellurics if both exist
    if telluric_info_a[1]!= telluric_info_b[1]:
        telluric_spec = (telluric_spec_a*(target_info_b[3] - telluric_info_b[3]) + telluric_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3])
        telluric_err_spec = (((telluric_err_spec_a*(target_info_b[3] - telluric_info_b[3]))/(telluric_info_a[3]-telluric_info_b[3]))**2 + ((telluric_err_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3]))**2)**0.5
        
    else:
        telluric_spec = telluric_spec_a
        telluric_err_spec = telluric_err_spec_a    
    
    
    telluric = np.zeros([np.shape(all_log_w)[0],np.shape(all_log_w)[1]])
    telluric_error = np.zeros([np.shape(all_log_w)[0],np.shape(all_log_w)[1]])
    
        
    for fibre in range(19):
        for order in range(40):
            # interpolate the telluric spectrum onto that of the observation        
            telluric_interpolation_func = InterpolatedUnivariateSpline(wave_tell_a[:,order], telluric_spec[:,order],k=1)
            telluric_err_interpolation_func = InterpolatedUnivariateSpline(wave_tell_a[:,order], telluric_err_spec[:,order],k=1)
            
            telluric[:,order] = telluric_interpolation_func(all_log_w[:,order,fibre])
            telluric_error[:,order] = telluric_err_interpolation_func(all_log_w[:,order,fibre])
            
            for wave in range(np.shape(all_log_w)[0]):
                all_s_logerrflux[wave,order,fibre] = ((all_s_logerrflux[wave,order,fibre]/all_s_logflux[wave,order,fibre])**2 + (telluric_error[wave,order]/telluric[wave,order])**2)**0.5 

            all_s_logflux[:,order,fibre] /= telluric[:,order]
            
            all_s_logerrflux[:,order,fibre] *= all_s_logflux[:,order,fibre]
    
    if not os.path.exists('/priv/avatar/ehold13/twmn_tellurics/'+obs_fits[0:10]+'_'+date+'.fits'):
        prim = pyfits.PrimaryHDU()
        tel = pyfits.ImageHDU(telluric,name = 'telluric_spect')
        waves = pyfits.ImageHDU(all_log_w[:,:,-1], name = 'wavelength')
        tel_err = pyfits.ImageHDU(telluric_error,name = 'telluric_err_spect')  
        
        hdul = pyfits.HDUList([prim, tel, waves,tel_err])
        hdul.writeto('/priv/avatar/ehold13/twmn_tellurics/'+obs_fits[0:10]+'_'+date+'.fits')
          
    
    wavelength = all_log_w
    spect = all_s_logflux
    spect_err = all_s_logerrflux

    if combine_fibres:

        spect = np.zeros((np.shape(wavelength)[0],40))
        spect_err = np.zeros((np.shape(wavelength)[0],40))
        for fibre in range(19):
            for order in range(40):
                spect[:,order] += all_s_logflux[:,order,fibre]
                spect_err[:,order] += all_s_logerrflux[:,order,fibre]
    prim = pyfits.PrimaryHDU()
    ph = prim.header 
    ph.set('Fits', obs_fits)
    ph.set('Date', date)   
    primary_hdu = pyfits.ImageHDU(spect, name = 'spectrum')
    image_hdu = pyfits.ImageHDU(wavelength, name= 'wavelength')
    image_hdu2 = pyfits.ImageHDU(spect_err, name = 'error')

    hdul = pyfits.HDUList([prim,primary_hdu, image_hdu, image_hdu2])
    hdul.writeto(save_dir+obs_fits[0:10] + '_corrected.fits')
    
    return spect, wavelength, spect_err
    
#c in km/s, in order to have reasonable scaling
c_km_s = c.c.to(u.km/u.s).value

def interp_template(lwave, template, lwave0, dlwave, deriv=False):
    """Interpolate at template spectrum evenly sampled in log(wavelength), 
    at a grid of log(wave) points

    Parameters
    ----------
    lwave: numpy array
        (natural) logarithm of wavelength in Angstroms

    template: numpy array
        template spectrum

    lwave0: (natural) logarithm of the first template wavelength
    dlwave: (natural) logarithm spacing of the template grid

    Returns
    -------
    Either the interpolated spectrum, or the derivative with respect to the 
    relativistic factor (~v/c), which is the same as the derivative 
    with respect to log(wave) if deriv=True
    """
    ix = (lwave - lwave0)/dlwave
    #Don't go past the edges.
    if (np.min(ix) < 0) or (np.max(ix) > len(template)-1):
        raise UserWarning("Input wavelength outside range!")
    #ix_int = np.maximum(np.minimum(ix.astype(int), len(template)-2), 0)
    #frac = np.maximum(np.minimum(ix - ix_int, 1), 0)
    ix_int = ix.astype(int)
    frac = ix - ix_int
    if deriv:
        #The derivative of the return line below with respect to frac, divided by dlwave
        return (template[ix_int+1] - template[ix_int])/dlwave
    else:
        return template[ix_int]*(1-frac) + template[ix_int+1]*frac


def rv_jac_old(params, wave, spect, spect_err, interp_func,vo = 0, ve = 0):
    """This was Erin's original version of rv_jac. or some reason, it didn't quite
    function... maybe because of the error in scaling_factor"""
    pixel = (wave-0.5*(wave[0]+wave[-1]))/(wave[-1]-wave[0])
    jac = np.zeros([len(pixel),4])
    
    scaling_factor = np.exp((params[1] + params[2]*pixel*(params[3]*pixel)))
    relativistic_factor = (1+vo/c_km_s)/(1+ve/c_km_s)
    
    fitted_spect = interp_func(relativistic_factor* wave * (1.0 + params[0]/c_km_s))*scaling_factor
    
    jac[:,0] = (interp_func(relativistic_factor*wave*(1.0 + (params[0] + 1e-6)/c_km_s))*scaling_factor - fitted_spect)/(1e-6 * spect_err)
    jac[:,1] = fitted_spect/spect_err
    jac[:,2] = pixel * (fitted_spect/spect_err)
    jac[:,3] = pixel*pixel * (fitted_spect/spect_err)
    
    return jac
    
def rv_fitting_eqn_old(params, wave, spect, spect_err, interp_func, return_spec = False):
    """RV fitting based on a UnivariateSpline interpolation function
    """
    #print(params) #This can be used as a check...
    pixel = (wave-0.5*(wave[0]+wave[-1]))/(wave[-1]-wave[0])

    scaling_factor = np.exp(params[1] + params[2]*pixel + params[3]*pixel**2)
    
    beta = params[0]/c_km_s
    relativistic_factor = np.sqrt( (1+beta)/(1-beta) )
    
    fitted_spectra = interp_func(relativistic_factor * wave )*scaling_factor
    
    if return_spec:
        return fitted_spectra
    return (fitted_spectra - spect)/spect_err
    
def rv_fitting_eqn(params, lwave, spect, spect_err, template, lwave0, dlwave, return_spec = False):
    """Calculate the resicual vector for RV fitting. 

    Parameters
    ----------
    params: numpy array
        rv in in km/s, multiplier, linear and parabolic spectral slope.
    lwave: numpy array
        Logarithm of wavelength in Angstroms for target spectrum.
    spect: numpy array
        target spectrum
    spect_err: numpy array
        uuncertainties in the target spectrum
    template: numpy array
        Template on a logarithmic grid.
    lwave0: logarithm of the first element of the template's wavelength
    dlwave: logarithmic step between wavelengths
    return_spec: bool
        Set to true to return the fitted spectrum.

    Returns
    -------
    out: numpy array
        Either the residual vector of fitted spectrum (see return_spec)
    """
    #print(params) #This can be used as a check...
    pixel = (lwave-0.5*(lwave[0]+lwave[-1]))/(lwave[-1]-lwave[0])

    scaling_factor = np.exp(params[1] + params[2]*pixel+ params[3]*pixel**2)
    
    beta = params[0]/c_km_s
    relativistic_factor = np.sqrt( (1+beta)/(1-beta) )
    
    #fitted_spectra = interp_func(relativistic_factor * wave )*scaling_factor
    fitted_spectra = interp_template(lwave + np.log(relativistic_factor), template, lwave0, dlwave)*scaling_factor
    
    if return_spec:
        return fitted_spectra
    return (fitted_spectra - spect)/spect_err

def rv_jac(params, lwave, spect, spect_err, template, lwave0, dlwave):
    """Calculate the Jacobian for RV fitting. Requires 2 calles to interp_template (one of which is
    likely redundant if least_squares also calles rv_fitting_eqn).

    Parameters
    ----------
    params: numpy array
        rv in in km/s, multiplier, linear and parabolic spectral slope.
    lwave: numpy array
        Logarithm of wavelength in Angstroms for target spectrum.
    spect: numpy array
        target spectrum
    spect_err: numpy array
        uuncertainties in the target spectrum
    template: numpy array
        Template on a logarithmic grid.
    lwave0: logarithm of the first element of the template's wavelength
    dlwave: logarithmic step between wavelengths

    Returns
    -------
    jac: numpy array
        spectrum_length x 4 Jacobian.
    """
    pixel = (lwave-0.5*(lwave[0]+lwave[-1]))/(lwave[-1]-lwave[0])
    jac = np.zeros([len(pixel),4])
    
    scaling_factor = np.exp(params[1] + params[2]*pixel+ params[3]*pixel**2)
    beta = params[0]/c_km_s
    relativistic_factor = np.sqrt( (1+beta)/(1-beta) )
    
    fitted_spect = interp_template(lwave + np.log(relativistic_factor), template, lwave0, dlwave)*scaling_factor
    
    jac[:,0] = interp_template(lwave + np.log(relativistic_factor), template, lwave0, dlwave, deriv=True)/c_km_s*scaling_factor/spect_err
    jac[:,1] = fitted_spect/spect_err
    jac[:,2] = pixel * (fitted_spect/spect_err)
    jac[:,3] = pixel*pixel * (fitted_spect/spect_err)
    
    return jac

def test_rv_chi2(params, rvs, lwave, spect, spect_err, template, lwave0, dlwave):
    """
    Test the RV chi^2 curve as a function of RV.
    """
    chi2s = np.empty_like(rvs)
    for i in range(len(rvs)):
        thisparams = params.copy()
        thisparams[0] = rvs[i]
        resid = rv_fitting_eqn(thisparams, lwave, spect, spect_err, template, lwave0, dlwave)
        chi2s[i] = np.sum(resid**2)
    return chi2s

def fibre_velocity_adjustment():
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
    good_orders_mn = np.empty((len(rvs[:,0,0]),len(rvs[0,0,:])))
    good_orders_err = np.empty((len(rvs[:,0,0]),len(rvs[0,0,:])))
    
    #good_orders_mn = np.mean(rvs[:,30:36,:], axis=1)
    #good_orders_err = np.mean(rv_errs[:,30:36,:], axis=1) 

    for fibre in range(19):
        weights = 1/rv_errs[:,3:,fibre]**2
        weights = np.where(np.isinf(weights),0,weights)
        good_orders_mn[:,fibre] = np.nansum(weights * rvs[:,3:,fibre],axis=1)/np.nansum(weights, axis = 1)
        good_orders_err[:,fibre] = 1/(np.nansum(weights,axis=1))**0.5
    
    #Look for fiber-to-fiber outliers
    fiber_std = np.nanstd(good_orders_mn, axis=1)
    straight_mean_rv = np.nanmean(good_orders_mn, axis=1)
    weighted_mean_rv = np.nansum(good_orders_mn/good_orders_err**2, axis=1)/np.nansum(1/good_orders_err**2, axis=1)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    good_obs = (fiber_std < 3*np.median(fiber_std)) & (np.abs(weighted_mean_rv) < 0.08)
    #good_obs = (fiber_std < 3*np.median(fiber_std))
   

    #plt.imshow(good_orders_mn[good_obs], aspect='auto', interpolation='nearest')

    fiber_rvs = good_orders_mn.copy()
    for ix,obs_rv  in enumerate(fiber_rvs):
        obs_rv -= weighted_mean_rv[ix]

    fiber_means = np.mean(fiber_rvs[good_obs], axis=0)
    fiber_std = np.std(fiber_rvs[good_obs], axis=0)

    #Lets correct for this and see if it helps.By only subtracting fiber-to-fiber means, this should 
    #not bias the radial velocity observations. To check!s
    corrected_gd_mn = good_orders_mn
    corrected_gd_mn -= fiber_means
    corrected_wt_mn = np.nansum(corrected_gd_mn/good_orders_err**2, axis=1)/np.nansum(1/good_orders_err**2, axis=1)
    
    return fiber_means
    

def combination_method_two(observation_dir = '/home/ehold13/veloce_scripts/veloce_reduction/10700/', dispersion_limit = 1):
    """
    Description
    -----------
    This function will compute the mean square residual for all files in the observation_dir
    
    Parameters
    ----------
    observation_dir : type - string
        The folder where the files are to compute the mean square residual for each order_index
    
    dispersion_limit : type - float64
        The dispersion between fibres that will be tolerated
    
    Returns
    -------
    mean_sq_resid : type - numpy nd-array
        The mean square residuals for each of the orders
    """
    count = 0
    #all_obs_rvs = []
    all_order_rvs = []
    all_order_rvs_err = []
    fib_adj = fibre_velocity_adjustment()
    
    fit_rv = np.empty(40)
    fit_rv_err = np.empty(40)

    # iterate over the Tau Ceti Observations
    for fit_index,fits in enumerate(os.listdir(observation_dir)):
        if fits.endswith('.fits'):  # fits in ['fits_191211.fits']:  #fits.endswith('.fits'):
            observations = pyfits.open(observation_dir + fits)
            
            # now take weighted mean over the fibres to get a velocity per order for each observation
            order_rv = np.empty((len(observations['RV'].data[:,0,0]),len(observations['RV'].data[0,:,0])))
            
            order_rv_err = np.empty((len(observations['RV'].data[:,0,0]),len(observations['RV'].data[0,:,0])))
       
            # for each observation on this date check to see if have low dispersion (i.e. is a good observation)
            for obs in range(len(observations['RV'].data[:,0,0])):
                rvs = observations['RV'].data[obs,:,:]
                for i in range(19):
                            rvs[:,i] -= fib_adj[i]
                            
                errors = observations['ERROR'].data[obs,:,:]
                
                if np.std(rvs[30:36,:]) > dispersion_limit:
                    order_rv[obs,:] = np.nan
                    order_rv_err[obs,:] = np.nan
                else:
                    if True: 
                        
                        count += 1
                        

                        # combine fibres with weighted-mean
                        for order in range(len(rvs[:,0])):
                            weights =  1/errors[order,:]**2
                            
                            for i,weight in enumerate(weights):
                                if np.isinf(weight):
                                    weights[i] = 0
                            
                            if np.nansum(weights) == 0:
                                order_rv[obs,order] = np.nan
                                order_rv_err[obs,order] = np.nan
                            else:
                                order_rv[obs,order] = np.nansum(weights*rvs[order,:])/np.nansum(weights)
                                order_rv_err[obs,order] = 1/np.sqrt(np.nansum(weights))
                                               
            all_order_rvs.extend(order_rv)
            all_order_rvs_err.extend(order_rv_err)
            
    all_order_rvs = np.array(all_order_rvs)
    all_order_rvs_err = np.array(all_order_rvs_err)
        
    
    for ords in range(40):
        weights = 1/all_order_rvs_err[:,ords]**2
                   
        for i, weight in enumerate(weights):
            if np.isinf(weight):
                weights[i] = 0
                
        if np.nansum(abs(weights))==0:
            fit_rv[ords] = np.nan
            fit_rv_err[ords] = np.nan
        else:
            fit_rv[ords] = np.nansum(weights*all_order_rvs[:,ords])/np.nansum(weights)
            fit_rv_err[ords] = 1/np.sqrt(np.nansum(abs(weights)))
    
    all_obs_rvs = fit_rv
    all_obs_rvs = np.array(all_obs_rvs)
   
    
    all_obs_rvs = np.where(np.isinf(all_obs_rvs),np.nan,all_obs_rvs)
    
    all_order_rvs= np.where(np.isinf(all_order_rvs),np.nan,all_order_rvs)
   
    mean_sq_resid = np.nanmedian((all_obs_rvs - all_order_rvs)**2,0)  
    
    
     
    mean_sq_resid = np.where(np.isnan(mean_sq_resid), 1,mean_sq_resid)
    
    
    return mean_sq_resid
            
    
def PCA_combination(observation_dir, p = None):
    """
    Description
    -----------
    
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # get the systemtic error for each order
    combination = combination_method_two()
    
    # initiate lists
    median_flux = []
    num_obs_each_day = []
    v_all = []
    v_all_err = []
    v_days = []
    jds = []
    files = []
    # extract the velocities
    for fits in os.listdir(observation_dir):
        if fits.endswith('.fits'):
            
            observations = pyfits.open(observation_dir + fits)
            num_obs_each_day.append(len(observations['RV'].data[:,0,0]))
            jds.extend(observations[4].data['mjds'])

            for obs in range(len(observations['RV'].data[:,0,0])):
                # here can limit to only include some observations if wish
                if True:
                    files.append(fits)
                    median_flux.append(observations['median_flux'].data[obs,:,:])
                    v_all.append(observations['RV'].data[obs,:,:])
                    v_all_err.append(observations['ERROR'].data[obs,:,:])
            # collect all rvs for a given day        
            v_days.append((len(observations['RV'].data[:,0,0]),observations['RV'].data,observations['ERROR'].data))
    
    # change lists to be arrays
    median_flux = np.array(median_flux)
    v_all = np.array(v_all)
    v_all_err = np.array(v_all_err)

    # add in the systematic errors
    for obs in range(len(v_all_err[:,0,0])):
        for ords in range(len(v_all_err[0,:,0])):
            for fib in range(len(v_all_err[0,0,:])):
                
                q = np.nanmax([combination[ords] - v_all_err[obs,ords,fib]**2,0])**0.5
                
                if q > 0.02:
                    q = np.inf
                
                v_all_err[obs,ords,fib] = (q**2 + v_all_err[obs,ords,fib]**2)**0.5
    
    # get the mean flux for each order    
    order_means = np.mean(np.mean(median_flux,0),1)
    
    # only use orders which have no NaN values for any observation
    good_orders = np.where(order_means == order_means)[0] 
    
    # restrict the data to only include the good orders
    median_flux = median_flux[:,good_orders,:]
    v_all = v_all[:,good_orders,:]
    v_all_err = v_all_err[:,good_orders,:]
    
    
    # calculate the wtmn velocity for each day of observations    
    v_day_wtmn = [] 
   
    for elem in v_days:
        vs = elem[1]
        vserr = elem[2]
        ws = 1/vserr**2
        
        ws = np.where(np.isinf(ws),0,ws)
        
                
        v_day_wtmn.extend([np.nansum(ws*vs)/np.nansum(ws)]*elem[0])
                
    v_day_wtmn = np.array(v_day_wtmn)
    #import pdb; pdb.set_trace()
    
    
    v_wtmn = np.empty(len(v_all)) 
    v_wtmn_err = np.zeros(len(v_all))
    
    # initiate lists with shape observations x (orders * fibres)
    v = v_all.reshape((v_all.shape[0],v_all.shape[1]*v_all.shape[2]))
    v_err = v_all_err.reshape((v_all_err.shape[0],v_all_err.shape[1]*v_all_err.shape[2]))
    
    # calculate the weighted mean for each observation 
    for obs in range(len(v)):
        w = 1/v_err[obs]**2
        
        for i,weight in enumerate(w):
                            if np.isinf(weight):
                                w[i] = 0
        v_wtmn[obs] = np.nansum(w*v[obs])/np.nansum(w)
        v_wtmn_err[obs] = 1/np.sqrt(np.nansum(abs(w)))
    
    
    for obs in range(np.shape(median_flux[:,0,0])[0]):
        for order in range(np.shape(median_flux[0,:,0])[0]):
            median_flux[obs,order,:] /= np.nanmean(median_flux[obs,order,:])
            
    # reshape the flux array to be observations x (orders * fibres)   
    X = median_flux.reshape((median_flux.shape[0],median_flux.shape[1]*median_flux.shape[2]))
                 
    # calculate the eigenvalues and eigenvectors for X, find structures in the flux
    W,V = np.linalg.eigh(np.dot(X.T,X))
    
    # plot the eigenvalues
    if True:
        plt.figure()
        plt.semilogy(W,'ko')
        plt.xlabel('Velocity Number')
        plt.ylabel('Eigenvalue')
        plt.show()
    
    # observed that the dominant eigenvalues were the last four, so save the corresponding eigenvectors
    A = V[:,-4:]
    
    # calculate 
    Y = np.dot(X,A)
    
    # if not given initial values for the fibre structure, calculate it
    if p == None:
        p = np.dot(np.dot(np.linalg.inv(np.dot(Y.T,Y)),Y.T),v_wtmn)
    
    # calculate the adjusted velocities
    v_adjust = v_wtmn - np.dot(Y,p)
    
    # for each day of observations, calculate the velocity
    
    # initiate lists
    v_day = []
    v_day_err = []
    mjds_day = []
    
    # make list of lists, where each sublist is all of the velocities for that day
    for i,elem in enumerate(num_obs_each_day):
        if i != 0:
            v_day.append([v_adjust[sum(num_obs_each_day[0:i]):sum(num_obs_each_day[0:i+1])]])
            v_day_err.append([v_wtmn_err[sum(num_obs_each_day[0:i]):sum(num_obs_each_day[0:i+1])]])
            mjds_day.append(np.nanmean(jds[sum(num_obs_each_day[0:i]):sum(num_obs_each_day[0:i+1])]))
        else:
            v_day.append([v_adjust[0:elem]])
            v_day_err.append([v_wtmn_err[0:elem]])
            mjds_day.append(np.nanmean(jds[0:elem]))
    
    # take weighed mean over each day of velocities
    v_day1 = []
    v_day_err1 = []
    for i,elem in enumerate(v_day):
        w = 1/np.array(v_day_err[i])**2
        w = np.where(np.isinf(w),0,w)
        
        v_day1.append(np.nansum(w * np.array(elem))/np.nansum(w))
        v_day_err1.append(1/np.nansum(w)**0.5)
     
    v_day1 = np.array(v_day1)
    # plot the daily velocities and calculate the RMS
    if True:
        plt.figure()
        plt.errorbar(Time(mjds_day, format='mjd').to_datetime(),(v_day1-np.nanmean(v_day1))*1000,yerr = np.array(v_day_err1)*1000,fmt = 'ko',capsize=5)
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Observation Date')
        plt.show()
        
        print((np.nansum((v_day1-np.nanmean(v_day1))**2)/len(v_day1))**0.5 * 1000)
        print('rms, daily: ',np.std((v_day1-np.nanmean(v_day1)))*1000)
    #import pdb; pdb.set_trace()
    
    # plot the velocity for each observation against each mode
    if False:
        plt.figure()
        plt.plot(Y[:,0],v_wtmn,'.')
        plt.title('First Mode')
        plt.ylabel('V_wtmn')
        plt.xlabel('Y[:,0]')
        
        plt.figure()
        plt.plot(Y[:,1],v_wtmn,'.')
        plt.title('Second Mode')
        plt.ylabel('V_wtmn')
        plt.xlabel('Y[:,1]')
        
        
        plt.figure()
        plt.plot(Y[:,2],v_wtmn,'.')
        plt.title('Third Mode')
        plt.ylabel('V_wtmn')
        plt.xlabel('Y[:,2]')
        
        
        plt.figure()
        plt.plot(Y[:,3],v_wtmn,'.')
        plt.title('Fourth Mode')
        plt.ylabel('V_wtmn')
        plt.xlabel('Y[:,3]')
        
        plt.show()
    
    
    # calculate the RMS for all observations and plot the data
    if True:
        mn = 0
        rms = 0
        count = 0
        # just to remove observations that are very wrong velocity
        v_adjust = np.where(abs(v_adjust)<0.08,v_adjust,np.nan)
        plt.figure()
        for i in range(len(v_adjust)):
            if not np.isnan(v_adjust[i]):
                rms += (v_adjust[i] - np.nanmean(v_adjust))**2
                count += 1
                
                plt.errorbar([Time(jds[i], format = 'mjd').to_datetime()],(v_adjust[i] - np.nanmean(v_adjust))*1000,yerr = v_wtmn_err[i]*1000,fmt = 'k.',capsize=5)
                
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()
        
        print('rms all obs',(rms/count)**0.5 * 1000)
    return jds, v_adjust, v_wtmn_err, files, p
              

def generate_rvs(star_name, date, template_path, int_guess = 0.01, alpha = 0.2, residual_limit = 0.5,runs = 1, total_runs = 5):
    """
    Description
    -----------
    This function will generate the radial velocities for each fibre and order.
    
    Parameters
    ----------
    star_name : type - string
        Name of the star calculating the velocity for
    
    date : type - string
        Date of the observations in yymmdd
    
    template_path : type - string
        The path to the template used to calculate velocities
    
    int_guess : type - float64
        Initial velocity guess for radial velocity in km/s
    
    alpha : type - float64
        Value to multiply log(telluric depth) by to add to errors to weight points less
    
    residual_limit : type - float64
        Residual limit tolerate for a point to be removed to re-run fitting
    
    runs : type - int
        current run number of velocity fit
    
    total_runs : type - int
        The total number of runs to do removing points where fit is bad.
    
    Returns
    -------
    velocity_errors : type - numpy nd-arrays
       Measure of different between velocity and barycentric velocity for each file and order, use if testing data quality and have done no corrections
    
    files : type - list
        List of files where velocity has been calculated
    
    orders : type - list
        All of the orders where the velocity was calculated   
    
    """   
    
    veloce_obs = Table.read('/home/ehold13/veloce_scripts/veloce_observations.fits')

    # limit to the given star name and date
    stars = veloce_obs[veloce_obs['star_names']==star_name]
    
    obs_index = []

    for j,star in enumerate(stars):
        for index,yymmdd in enumerate(star[8]):
            if yymmdd.decode('utf-8') == date:
                obs_index.append((j,index))

    files = [stars[obs[0]][7][obs[1]].decode('utf-8') for obs in obs_index]
    
    f = [str(files[i]) for i in range(len(files))] 

    orders = list(range(40)) 
    #orders = [6,7,13,14,17,25,26,27,28,30,31,33,34,35,36,37]
    rvs = np.zeros((len(files),len(orders),19))
    rv_errs = np.zeros((len(files),len(orders),19))
    mses = np.zeros((len(files),len(orders),19))
    med_flux = np.zeros((len(files),len(orders),19))

    wtmn_rv = np.empty((len(files),len(orders)))
    wtmn_rv_err = np.empty((len(files),len(orders)))
    velocity_errors = np.empty((len(files),len(orders)))
    total_rv = np.empty(len(files))
    total_rv_error = np.empty(len(files))
    
    BCs = np.empty(len(files))
    MJDs = np.empty(len(files))
    
    
    for fit_index,fits in enumerate(files):
        bar = ShadyBar('Fitting', max=len(orders)*19, suffix = '%(percent).1f%%, time remaining %(eta)ds')
        
        obs_file_path = '/priv/avatar/ehold13/obs_corrected/'+star_name+'/'+fits[0:10]+'_corrected.fits'
        observation = pyfits.open(obs_file_path)
        
        orig_file = pyfits.getheader('/priv/avatar/velocedata/Data/spec_211202/' + date + '/' + fits[0:10] + 'oi_extf.fits')
        MJDs[fit_index] = orig_file['UTMJD']
        
        BC_t, BC_star = barycentric_correction('11dec30096o.fits',fits[0:10]+'o.fits','191211',date)
        BCs[fit_index] = BC_star*c.c.to(u.km*u.s**-1).value
                
        print('\nFitting Observation '+fits+', '+str(int(fit_index)+1)+'/'+str(len(files)))
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #BC_star = 0
        spectrum = observation['spectrum'].data
        wavelength = observation['wavelength'].data
        wavelength += BC_star * wavelength
        error = observation['error'].data
        
        
        for order_index,order in enumerate(orders): #enumerate(range(np.shape(spectrum)[1])):
            
            tellurics = pyfits.open('/home/ehold13/veloce_scripts/tellurics_lines_widths_depths.fits')
            lines = tellurics[0].data[order_index,:]
            mask = np.isnan(lines)
            lines = lines[~mask]
            widths = tellurics[1].data[order_index,:][~mask]
            depths = tellurics[2].data[order_index,:][~mask]

            
            template = pyfits.open(template_path)
            temp_wave = template[1].data[:,order]
            temp_spec = template[0].data[:,order]
            gaus = np.exp(-np.linspace(-2,2,15)**2/2) 
            gaus /= np.sum(gaus)
            temp_spec = np.convolve(temp_spec,gaus, mode='same')
            
            temp_func = InterpolatedUnivariateSpline(temp_wave, temp_spec, k=1) 
            temp_lwave = np.log(temp_wave)
            
            temp_dlwave = temp_lwave[1]-temp_lwave[0]
            
            
            for fibre_index,fibre in enumerate(range(np.shape(spectrum)[2])):
                try:
                    runs = 1
                    spect = spectrum[830:3200,order,fibre]
                    mask = np.isnan(spect)
                    spect = spect[~mask]
                    wave = wavelength[830:3200,order,fibre][~mask].astype(np.float64)
                    log_wave = np.log(wave)
                    err = error[830:3200,order,fibre][~mask]
                    


                    med_flux[fit_index,order_index,fibre_index] = np.median(spect)
                    
                    mask = np.isnan(spect)
                    scale = np.median(spect[~mask])
                    spect /= scale
                    err /= scale
                    scaled_median = med_flux[fit_index,order_index,fibre_index]/scale
                    
                 
                    if 0==len(log_wave):
                        bar.next()
                        continue
                        
                    for index, line in enumerate(lines):       
                       err = np.where((line - widths[index] < wave) & (wave < line + widths[index]),-alpha*np.log(abs(spect)),err)
                       err = np.where(spect>3*abs(scaled_median),np.inf,err)
                        
       
                    
                    while runs <= total_runs:    
                        if np.isinf(err).all():
                            bar.next()
                            continue    
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
                        #if BCs[fit_index] >0:
                        #    initial_cond = [10,0,0,0] 
                        #else:
                        #    initial_cond = [-10,0,0,0] 
                        initial_cond = [int_guess,0,0,0]
                        a = optimise.least_squares(rv_fitting_eqn,x0 = initial_cond, args=(log_wave, spect, err, temp_spec, temp_lwave[0], temp_dlwave), \
                            jac=rv_jac, method='lm')
                        
                        for i,value in enumerate(a.fun):
                            if abs(value) > residual_limit:
                                err[i] = np.inf
                        runs += 1
                    
                      
                           
                    if a.success:
                        try:
                            cov = np.linalg.inv(np.dot(a.jac.T,a.jac))  
                            rvs[fit_index,order_index,fibre_index] = a.x[0]
                            mse = np.mean(a.fun**2) 
                            mses[fit_index,order_index,fibre_index] = mse
                            rv_errs[fit_index,order_index,fibre_index] = np.sqrt(mse)*np.sqrt(cov[0,0])
                        except:
                            print('Singular Matrix')
                    bar.next()    
                except ValueError:
                    #print('Infinite Error - cannot fit this order')
                    bar.next()
                except UserWarning:
                    #print('Wavelength outside of wavelength range - cannot fit this order')
                    bar.next()
            weights = 1/rv_errs[fit_index,order_index,:]**2
            wtmn_rv[fit_index,order_index] = np.nansum(weights*rvs[fit_index,order_index,:])/np.nansum(weights)
            wtmn_rv_err[fit_index,order_index] = 1/np.sqrt(np.nansum(weights))
            
                
            #print(order_index,wtmn_rv[fit_index,order_index]*1000,'+/-',wtmn_rv_err[fit_index,order_index]*1000, 'm/s')
            
            velocity_errors[fit_index, order_index] = (wtmn_rv[fit_index,order_index]*1000 - BC_star*c.c.to(u.m/u.s).value)
        bar.finish()
        total_weights = 1/wtmn_rv_err[fit_index,:]**2
        for i,elem in enumerate(total_weights):
            if np.isinf(elem):
                total_weights[i] = 0
        total_rv[fit_index] = np.nansum(total_weights*wtmn_rv[fit_index,:])/np.nansum(total_weights)
        total_rv_error[fit_index] = 1/np.sqrt(np.nansum(total_weights))
        print('final velocity for observation', total_rv[fit_index]*1000, '+/-',total_rv_error[fit_index]*1000, 'm/s')
            
    final_weights = 1/total_rv_error**2
    
    for i, elem in enumerate(final_weights):
        if np.isinf(elem):
            final_weights[i] = 0
    final_rv = np.nansum(final_weights * total_rv)/np.nansum(final_weights)
    final_error = 1/np.sqrt(np.nansum(final_weights))
    print('Final Velocity (m/s): ',final_rv*1000,' +/- ', final_error*1000)
    
    print('Velocity uncertainty, list of orders (m/s): {:.1f}'.format(np.std(np.mean(velocity_errors, axis=1)))) # was 24:34
    print('Internal dispersion, based on scatter between orders (m/s): ')
    simple_std = np.std(velocity_errors, axis=1)/np.sqrt(len(orders))
    print(simple_std) 
    
    if not os.path.exists('/home/ehold13/veloce_scripts/veloce_reduction/'+star_name+'/fits_'+str(date)+'.fits'):
 
        c1 = pyfits.Column(name ='Template',format=str(len(template_path))+'A',array = [template_path])
        c2 = pyfits.Column(name = 'Files',format = str(len(files[0]))+'A',array = files)
        c3 = pyfits.Column(name = 'MJDs', format = '1D',array = MJDs)
        c4 = pyfits.Column(name = 'BCs', format = '1D', array = BCs)
        coldefs = pyfits.ColDefs([c1, c2, c3, c4])
        
        table_hdu = pyfits.BinTableHDU.from_columns(coldefs)
        primary_hdu = pyfits.PrimaryHDU()
        primary_header = primary_hdu.header
        primary_header.set('Star', star_name)
        primary_header.set('Date', date)
        image_hdu2 = pyfits.ImageHDU(rvs, name = 'rv')
        image_hdu3 = pyfits.ImageHDU(rv_errs, name = 'error')
        image_hdu4 = pyfits.ImageHDU(med_flux, name = 'median_flux')
        
        hdul = pyfits.HDUList([primary_hdu,image_hdu2, image_hdu3, image_hdu4,table_hdu])
        hdul.writeto('/home/ehold13/veloce_scripts/veloce_reduction/'+star_name+'/fits_'+str(date)+'.fits') 
           
    return velocity_errors, files, orders
        
def rv_err_obs(file_names):
    """
    Description
    -----------
    Old function to check for self consistency.
    
    Parameters
    ----------
    
    Returns
    -------
    """
    rvs = []
    mnrvs = []
    times = []
    errs = []
    for ind, file_name in enumerate(file_names):
        dd = pyfits.open(file_name)
        wtmn_rv_ord = np.empty((len(dd[1].data),np.shape(dd[4].data)[1]))
        wtmn_rv_err_ord = np.empty((len(dd[1].data),np.shape(dd[4].data)[1]))
        
        wtmn_rv = np.empty(len(dd[1].data))
        wtmn_rv_err = np.empty(len(dd[1].data))
        
        for fit in range(len(dd[1].data)):
            for order in range(np.shape(dd[4].data)[1]):
                weights = 1/dd[5].data[fit,order,:]**2
                wtmn_rv_ord[fit,order] = np.sum(weights*dd[4].data[fit,order,:])/np.sum(weights)
                wtmn_rv_err_ord[fit,order] = 1/np.sqrt(np.sum(weights))
        
        for fit in range(len(dd[1].data)):
            weights = 1/wtmn_rv_err_ord[fit,:]**2
            wtmn_rv[fit] = np.sum(weights*wtmn_rv_ord[fit,:])/np.sum(weights)
            wtmn_rv_err[fit] = 1/np.sqrt(np.sum(weights))
        times.extend(Time(dd[2].data, format='mjd').to_datetime())
        rvs.extend(wtmn_rv*1000 - dd[3].data*c.c.value)
        mnrvs.extend([np.mean(wtmn_rv*1000 - dd[3].data*c.c.value)]*len(dd[3].data))
        errs.extend(wtmn_rv_err*1000)
    
    mn = 0
    rms = 0
    count = 0
    plt.figure()
    for i in range(len(errs)):
        if abs(np.array(rvs[i]) - np.array(mnrvs[i])) + errs[i] < 15:
            mn += rvs[i] - mnrvs[i]
            rms += (rvs[i] - mnrvs[i])**2
            count += 1
            plt.errorbar(times[i], rvs[i] - mnrvs[i],yerr = errs[i],fmt = 'k.',capsize=5)
    plt.ylabel('Barycentric Velocity Error (m/s)')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
    print('mean',mn/count)
    print('rms',(rms/count)**0.5)
    
def obs_creation_loop(star_name):
    """
    Description
    -----------
    Create all corrected fits files for a given star
    
    Parameters
    ----------
    star_name : type - string
        Name of the star want to create the corrected fits files for
    
    Returns
    -------
    None
    """
    dd = Table.read('/home/ehold13/veloce_scripts/veloce_observations.fits')
    
    stars = dd[dd['star_names']==star_name]
    for star in stars:
    
        for i, fit in enumerate(star[7]):
            if fit.decode('utf-8') != '':
                if not os.path.exists('/priv/avatar/ehold13/obs_corrected/'+star_name+'/'+fit.decode('utf-8')[0:10]+'_corrected.fits'):
                    print(fit)
                    create_observation_fits('11dec30096o.fits',fit.decode('utf-8'),star[8][i].decode('utf-8'),'/priv/avatar/ehold13/obs_corrected/'+star_name+'/')

def wtmn_combination(star_name, order_remove = []):
    """
    Description
    -----------
    Combine orders and fibres with a weighted-mean
    
    Parameters
    ----------
    star_name : type - string
        Name of the star to combine velocities
    
    order_remove : type - list
        Date, observation and the orders that need to be removed to fix a flagged point    
    
    
    Returns
    -------
    all_rvs : type - numpy nd-array
        Velocity for each observation
    
    day_rvs : type - numpy nd-array
        Velocity for each observation grouped by date
    """
    all_rvs = []
    day_rvs = []
    
    if len(order_remove) > 0:
        dates = [order_remove[i][0] for i in range(len(order_remove))]
        files = [order_remove[i][1] for i in range(len(order_remove))]
        orders = [order_remove[i][2] for i in range(len(order_remove))]
    
    for file_date in os.listdir('/home/ehold13/veloce_scripts/veloce_reduction/'+star_name+'/'):
        
        if file_date.endswith('.fits'):
            observations = pyfits.open('/home/ehold13/veloce_scripts/veloce_reduction/'+star_name+'/'+file_date)
            
            order_rv = np.empty((len(observations['RV'].data[:,0,0]),len(observations['RV'].data[0,:,0])))
            order_rv_err = np.empty((len(observations['RV'].data[:,0,0]),len(observations['RV'].data[0,:,0])))
            
            fit_rv = np.empty(len(observations['RV'].data[:,0,0]))
            fit_rv_err = np.empty(len(observations['RV'].data[:,0,0]))
            
            dispersion_date = []
            
            for obs in range(len(observations['RV'].data[:,0,0])):

                rvs = observations['RV'].data[obs,:,:]
                errors = observations['ERROR'].data[obs,:,:]

                errors = np.where(errors<10e-16,0,errors)

                # combine fibres with weighted-mean
                for order in range(len(rvs[:,0])):
                    weights =  1/errors[order,:]**2
                    for i,weight in enumerate(weights):
                        if np.isinf(weight):
                            weights[i] = 0
                            
                    if np.nansum(weights) < 10e-10 or order+65 in [67,69,79,80,95] or order not in range(30,36):
                        #order_rv[obs,order] = 0
                        order_rv_err[obs,order] = np.inf
                    # remove orders which user has flagged as bad
                    elif file_date[5:11] in dates and obs in files[dates.index(file_date[5:11])] and order in orders[dates.index(file_date[5:11])][obs]:
                        #order_rv[obs,order] = 0
                        order_rv_err[obs,order] = np.inf
                    else:
                        order_rv[obs,order] = np.nansum(weights*rvs[order,:])/np.nansum(weights)
                        order_rv_err[obs,order] = 1/np.sqrt(np.nansum(weights))
                    
                order_rv_err[obs,:] = np.where(order_rv_err[obs,:]<10e-16,0,order_rv_err[obs,:])
                # combine orders with weighted-mean
                weights = 1/order_rv_err[obs,:]**2

                for i,weight in enumerate(weights):
                    if np.isinf(weight):
                        weights[i] = 0
               
                if np.nansum(abs(weights))==0:
                    fit_rv[obs] = np.nan
                    fit_rv_err[obs] = np.nan
                
                else:
                    fit_rv[obs] = np.nansum(weights*order_rv[obs,:])/np.nansum(weights)
                    fit_rv_err[obs] = 1/np.sqrt(np.nansum(abs(weights)))
                    
                dispersion_date.append((observations[4].data['MJDs'][obs],fit_rv[obs],fit_rv_err[obs],(observations[4].data['Files'][obs],file_date[5:11])))
            
            disp_rvs = [dispersion_date[i][1] for i in range(len(dispersion_date))]
            
            if np.nanstd(disp_rvs) < 1:
                all_rvs.extend(dispersion_date)
                          
            weights = 1/fit_rv_err**2
            for i,weight in enumerate(weights):
                    if np.isinf(abs(weight)) or weight > 10e16:
                        weights[i] = 0
            if np.nansum(abs(weights))< 10e-16:
               rv = np.nan
               err = np.nan
            else:
                rv = np.nansum(weights*fit_rv)/np.nansum(weights)
                err = 1/np.sqrt(np.nansum(weights))
            if np.nanstd(disp_rvs) < 1:  
                #day_rvs.append((np.mean(observations[4].data['MJDs']),rv,err))
                day_rvs.append(dispersion_date)
    return all_rvs, day_rvs

def systematic_error_combination(star_name, order_remove=[]):
    """
    Description
    -----------
    Includes systematic errors in weighted-mean calculations
    
    Parameters
    ----------
    star_name : type - string
        Name of the star to calculate combined velocities of
    
    order_remove : type - list
        Date, observation and the orders that need to be removed to fix a flagged point    
    
    Returns
    -------
    all_rvs : type - string
        Velocity for each observation
    
    day_rvs : type - numpy nd-array
        Velocity for each observation grouped by date
    """
    all_rvs = []
    day_rvs = []
    combination = combination_method_two()
    fib_adj = fibre_velocity_adjustment()
    #print(fib_adj)
    if len(order_remove) > 0:
        dates = [order_remove[i][0] for i in range(len(order_remove))]
        files = [order_remove[i][1] for i in range(len(order_remove))]
        orders = [order_remove[i][2] for i in range(len(order_remove))]
     
    for file_date in os.listdir('/home/ehold13/veloce_scripts/veloce_reduction/'+star_name+'/'):
        
        if file_date.endswith('.fits'):
        
            observations = pyfits.open('/home/ehold13/veloce_scripts/veloce_reduction/'+star_name+'/'+file_date)
            
            
            order_rv = np.empty((len(observations['RV'].data[:,0,0]),len(observations['RV'].data[0,:,0])))
            order_rv_err = np.empty((len(observations['RV'].data[:,0,0]),len(observations['RV'].data[0,:,0])))
            
            fit_rv = np.empty(len(observations['RV'].data[:,0,0]))
            fit_rv_err = np.empty(len(observations['RV'].data[:,0,0]))
            
            q = np.empty(len(observations['RV'].data[0,:,0]))
            dispersion_date = []
            
            
            for obs in range(len(observations['RV'].data[:,0,0])):

                rvs = observations['RV'].data[obs,:,:]
                
                errors = observations['ERROR'].data[obs,:,:]
                
                
                for fib in range(19):
                    rvs[:,fib] -= fib_adj[fib]
                
                #rvs = np.delete(rvs,[4], axis = 1)
                #errors = np.delete(errors,[4], axis = 1)

                errors = np.where(errors<10e-16,0,errors)
                
                
                # combine fibres with weighted-mean
                for order in range(len(rvs[:,0])):
                    ord_std = np.nanstd(rvs[order,:])
                    ord_med = abs(np.nanmedian(rvs[order,:]))
                    errors[order,:] = np.where(abs(rvs[order,:])>ord_med+5*ord_std,np.inf,errors[order,:])
                    
                    weights =  1/errors[order,:]**2
                    for i,weight in enumerate(weights):
                        if np.isinf(weight):
                            weights[i] = 0
                    
                    if np.nansum(weights) < 10e-10 or order in [2]:
                        #order_rv[obs,order] = 0
                        order_rv_err[obs,order] = np.inf
                    # remove orders which user has flagged as bad
                    elif len(order_remove)>0 and file_date[5:11] in dates and obs in files[dates.index(file_date[5:11])] and order in orders[dates.index(file_date[5:11])][obs]:
                        #order_rv[obs,order] = 0
                        order_rv_err[obs,order] = np.inf
                    elif abs(np.nanmean(abs(rvs[order,:])))>np.nanmean(abs(rvs))+np.std(rvs)/4:
                       # order_rv[obs,order] = 0
                        order_rv_err[obs,order] = np.inf
                    
                    else:
                        order_rv[obs,order] = np.nansum(weights*rvs[order,:])/np.nansum(weights)
                        order_rv_err[obs,order] = 1/np.sqrt(np.nansum(weights))
                        q[order] = np.sqrt(np.nanmax([combination[order] - order_rv_err[obs,order]**2,0]))   
                
                for i, value in enumerate(q):
                    if value > 0.02:
                        q[i] = np.inf
                        
                order_rv_err[obs,:] = np.sqrt(order_rv_err[obs,:]**2 + q**2)
                 
                
                order_rv_err[obs,:] = np.where(order_rv_err[obs,:]<10e-16,0,order_rv_err[obs,:])
                
                rv_std = np.nanstd(order_rv[obs])
                rv_med = abs(np.nanmedian(order_rv[obs]))
                order_rv_err[obs,:] = np.where(abs(order_rv[obs])> 5*rv_std+rv_med, np.inf, order_rv_err[obs])
                
                
                # combine orders with weighted-mean
                weights = 1/order_rv_err[obs,:]**2
                
                for i,weight in enumerate(weights):
                    if np.isinf(weight):
                        weights[i] = 0
               
                if np.nansum(abs(weights)) < 10e-10:
                    fit_rv[obs] = np.nan
                    fit_rv_err[obs] = np.nan
                
                else:
                    fit_rv[obs] = np.nansum(weights*order_rv[obs,:])/np.nansum(weights)
                    fit_rv_err[obs] = 1/np.sqrt(np.nansum(abs(weights)))
                
                dispersion_date.append((observations[4].data['MJDs'][obs],fit_rv[obs],fit_rv_err[obs], (observations[4].data['Files'][obs],file_date[5:11])))
            
            disp_rvs = [dispersion_date[i][1] for i in range(len(dispersion_date))]
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!
            if np.nanstd(disp_rvs)<0.03:
                all_rvs.extend(dispersion_date) 
                
            else:
                std = np.std(disp_rvs)
                med = np.median(disp_rvs)
                for i,elem in enumerate(disp_rvs):
                    if abs(elem) > abs(med)+0.03:
                       
                       #dispersion_date[i] = (dispersion_date[i][0],np.nan, np.nan, dispersion_date[i][3])
                       del dispersion_date[i]
                       del disp_rvs[i]
                all_rvs.extend(dispersion_date)

            day_rvs.append(dispersion_date)
       
    return all_rvs, day_rvs

def func(params,x,y ,yerr,period,epoch,return_fit = False):
    """
    Description
    -----------
    General function to calculate velocity amplitude
    
    Parameters
    ----------
    params : type - list
        Fitting parameters
    x : type - numpy nd-array 
        Phase of each observation 
    y : type - numpy nd-array 
        Velocity of each point
    yerr : type - numpy nd-array
        Velocity error of each point 
    period : type - float 
        Period of the planet from TESS
    epoch : type - float64
        Epoch of the planet, from TESS
    return_fit : type - boolen (default - False)
        True if return velocity, False if return relative error
    
    Returns
    -------
    Relative error if return_fit False, velocity if return_fit True
    """
    
    if return_fit:
        return (params[0]*np.sin(2*np.pi*x+((epoch)%period)/period) +params[1])
    else:
        return (params[0]*np.sin(2*np.pi*x+(epoch%period)/period) +params[1] - y)/yerr
        
def linear_func(params,x,y,yerr, return_fit = False):
    """
    Description
    -----------
    Linear line function to use to correct for binary movement
    """
    
    if return_fit:
        return params[0]*x + params[1]
    else:
        return (params[0]*x + params[1] - y)/yerr
def mass(v,T,M_s,i, v_err, T_err, M_s_err,i_err):
    """
    Description
    -----------
    Calculate the mass of the planet in Earth masses.
    
    Parameters
    ----------
    v : type - astropy units
        velocity amplitue in m/s
    
    T : type - astropy units
        planet period in days
    
    M_s : type - astropy units
        star mass in solar masses
    
    i : type - float
        inclination in degrees
    
    v_err : type - astropy units
        velocity error in m/s
    
    T_err : type - astropy units
        period error in days
    
    M_s_err : type - astropy units
        star mass error in solar masses
    
    i_err : type - float
        inclination error in degrees
    
    Returns
    -------
    m : type - float 
        Mass of the planet in Earth masses
        
    m_err : type - float
        Error on planet mass in Earth masses
    """
    m = ((T/(2*np.pi*c.G))**(1/3) * abs(v) * M_s**(2/3))/np.sin(np.deg2rad(i))
    m_err = m  * ((i_err/np.tan(np.deg2rad(i)))**2 + (1/3 * T_err/T)**2 + (abs(v_err)/abs(v))**2 + (2/3 * M_s_err/M_s)**2)**0.5
    return m.to(u.M_earth), m_err.to(u.M_earth)
               
def flag_rvs(star_name, combination = 'systematic', plot = True, flagged_points = [], order_remove = []): 
    """
    Description
    -----------
    Go through each point and decide whether to flag it to check velocities
    
    Parameters
    ----------
    star_name : type - string
        Name of the star to check velocities of
        
    combination : type - string (default - 'systematic')
        the method desired to combine the order and fibre velocities, choose from 'wtmn', 'systematic'
        
    plot : type - boolean (default - True)
        If True, will fit velocity amplitude curve, else will show RMS
        
    flagged_points : type - list
        List of boolean values on whether to include each observation
        
    order_remove : type - list
        Date, observation and the orders that need to be removed to fix a flagged point    
    
    
    Returns
    -------
    flagged : type - list
        List of boolean values on whether to include each observation
    
    flagged_files : type - list
        List of flagged files
    
    xs1 : type - numpy nd-array
        MJDs for each of the observations
        
    
    ys1 : type - numpy nd-array
        Combined velocities for each observation
    
    yerr1 : type - numpy nd-array 
        Combined velocity errors for each observation
    
    files1 : type - numpy nd-array 
        List of files that have been combined
    
    day_rvs : type - numpy nd-array 
        All of the velocities for each observation grouped by date
    
    If plot = True also:
    
    inds : type - numpy nd-array 
        List of index to sort the observation in phase order
    """
    
    if plot:
        parameters = Table.read('known_parameters.csv')
        star_parameters = parameters[parameters['\ufeffname']==star_name][0]
        star_mass = star_parameters['star_mass']
        star_mass_error = star_parameters['star_mass_error']
        period = star_parameters['period']
        period_error = star_parameters['period_error']
        epoch = star_parameters['epoch']
        epoch_error = star_parameters['epoch_error']
        inclination = star_parameters['inclination']
        inclination_error = star_parameters['inclination_error']    
    
    # combine the order and fibre rvs depending on what the chosen combination method is
    if combination == 'wtmn':    
        all_rvs, day_rvs = wtmn_combination(star_name, order_remove)
        if plot:
            xs1 = [((all_rvs[i][0]-epoch)%period)/period for i in range(len(all_rvs))]
        else:
            xs1 = [all_rvs[i][0] for i in range(len(all_rvs))]
        ys1 = [all_rvs[i][1]*1000 for i in range(len(all_rvs))]
        yerr1 = [all_rvs[i][2]*1000 for i in range(len(all_rvs))]
        files1 = [all_rvs[i][3] for i in range(len(all_rvs))]
        
    elif combination == 'systematic':
        all_rvs, day_rvs = systematic_error_combination(star_name, order_remove)
        if plot:
            xs1 = [((all_rvs[i][0]-epoch)%period)/period for i in range(len(all_rvs))]
        else:
            xs1 = [all_rvs[i][0] for i in range(len(all_rvs))]
        
        ys1 = [all_rvs[i][1]*1000 for i in range(len(all_rvs))]
        yerr1 = [all_rvs[i][2]*1000 for i in range(len(all_rvs))]
        files1 = [all_rvs[i][3] for i in range(len(all_rvs))]
    elif combination == 'PCA':
        mjds, v, v_err, files, p = PCA_combination('/home/ehold13/veloce_scripts/veloce_reduction/'+star_name)
        xs1 = [((mjds[i]-epoch)%period)/period for i in range(len(mjds))]
        ys1 = [v[i]*1000 for i in range(len(v))]
        yerr1 = [v_err[i]*1000 for i in range(len(v))]
        files1 = [files[i] for i in range(len(files))]
        
    xs1 = np.array(xs1)
    ys1 = np.array(ys1)
    yerr1 = np.array(yerr1)
    files1 = np.array(files1)

    
    if plot: 
        # sort the points in order of increasing time
        inds = xs1.argsort()
        xs1 = xs1[inds]
        ys1 = ys1[inds]
        yerr1 = yerr1[inds]
        files1 = files1[inds]
            
        l_xs = len(xs1)
        orig_files = files1
        # remove the flagged points
        if len(flagged_points) != 0:
            xs2 = xs1[~np.array(flagged_points)]
            ys2 = ys1[~np.array(flagged_points)]
            yerr2 = yerr1[~np.array(flagged_points)]
            files2 = files1[~np.array(flagged_points)]   
        else:
            xs2 = xs1
            ys2 = ys1
            yerr2 = yerr1
            files2 = files1
        xs = []
        ys = []
        yerr = []
        files = []
        med = np.median(ys1)
        rej = []
        i = 0
        
        while i < len(ys2):
            ws = tk.Tk()
            ws.title('Keep Point?')
            if np.isnan(ys2[i]):
                ws.mainloop()
                i += 1
                continue
            k = np.where(abs(ys1-ys2[i])<1e-10)[0][0]
            
            val = ys1[i]
            
            keep = tk.BooleanVar(ws)
            keep.set(1)
            j = tk.IntVar(ws)  
            
            data1 = {'MJD': xs2[i:],
             'RV': ys2[i:]
            }
            
            df1 = DataFrame(data1,columns=['MJD','RV'])
        
        
            data2 = {'MJD': [xs2[i]],
             'RV': [ys2[i]]
            }
            
            df2 = DataFrame(data2,columns=['MJD','RV'])
            
            data3 = {'MJD': xs2[:i],
             'RV': ys2[:i]
            }
            
            df3 = DataFrame(data3,columns=['MJD','RV'])
            
            if len(rej)!= 0:
                data4 = {'MJD': xs1[np.array(rej)],
                 'RV': ys1[np.array(rej)]
                }
                
                df4 = DataFrame(data4,columns=['MJD','RV'])

            figure1 = plt.Figure(figsize=(10,7), dpi=100)
            ax1 = figure1.add_subplot(111)
            ax1.set_xlabel('Phase')
            ax1.set_ylabel('Velocity (m/s)')
            ax1.scatter(df1['MJD'],df1['RV'], color = 'k',label = 'To Do')
            ax1.scatter(df3['MJD'],df3['RV'], color = 'g', label = 'Accepted',marker = '^')
            if len(rej)!=0:
                ax1.scatter(df4['MJD'],df4['RV'], color = 'r', marker = 'X', s = 100, label = 'Flagged')
            ax1.scatter(df2['MJD'],df2['RV'], color = 'orange', marker = '*', s = 150, label = 'Current Point')
            ax1.legend()
            scatter1= FigureCanvasTkAgg(figure1, ws) 
            scatter1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
            
            b = tk.Button(ws,text = 'Keep', command =lambda:[keep.set(1),ws.destroy()]).pack()
            b = tk.Button(ws, text = 'Flag', command =lambda:[keep.set(0),ws.destroy()]).pack()
            b = tk.Button(ws,text = 'Keep Rest', command =lambda:[j.set(len(ys2)),ws.destroy()]).pack()
            
            text = tk.Text(ws)
           
            logs = np.array(glob.glob('/priv/avatar/velocedata/Data/Raw/'+str(files2[i][1])+'/ccd_3/[0-9,A-Z,a-z,_,-]*.log')).flatten()
            log = ''
            for l in logs:
                if 'vid.log' not in l:
                    log = l
            if log == '':
                log = logs[0]
                
            f = open(log,'r')
            log = f.readlines()
            
            veloce_obs = Table.read('/home/ehold13/veloce_scripts/veloce_observations.fits')
            star_obs = veloce_obs[veloce_obs['star_names']==star_name][0]
            
                      
            
            text1 = tk.Label(ws, text = 'File: '+files2[i][0]+', Date (yymmdd): '+files2[i][1]+', # Obs: '+str(np.count_nonzero(star_obs[8] == files2[i][1].encode('utf-8')))) 
            text1.configure(font=("Arial", 10, "bold"))
            text1.pack()
            
                
            text.insert(tk.INSERT, log) 
           
            text.pack(fill=tk.BOTH, expand=1)
            f.close()
            
            ws.mainloop()
            
            if keep.get(): 
                xs.append(xs2[i])
                ys.append(ys2[i])
                yerr.append(yerr2[i])
                files.append(files2[i])
            else:
                rej.append(k)
                xs.append(xs2[i])
                ys.append(ys2[i])
                yerr.append(yerr2[i])
                files.append(files2[i])
                
            if j.get()==len(ys2):
                xs.extend(xs2[i+1:])
                ys.extend(ys2[i+1:])
                yerr.extend(yerr2[i+1:])
                files.extend(files2[i+1:])
                i = j.get()
            i += 1
            
    else:
        inds = list(range(len(xs1)))
    if plot:
        if len(flagged_points)==0:
            flagged = [False]*l_xs
        else:
            flagged = flagged_points
        for point in range(l_xs):
            if point in rej:
                flagged[point] = True
                
        return flagged, np.array(orig_files)[np.array(flagged)], inds, xs1, ys1, yerr1, files1, day_rvs
    else:
        flagged = [False]*len(xs1)
        return flagged, np.array(list(range(len(xs1)))), xs1, ys1, yerr1, files1, day_rvs
               
def plot_phase(star_name, combination = 'systematic', plot = True, flagged_points = [], binary = False,order_remove = [], remove_dates = []):
    """
    Description
    -----------
    Plot the phase plot for given star
    
    Parameters
    ----------
    star_name : type - string
        Name of the star to plot phase of
        
    combination : type - string (default - 'systematic')
        How to combine the orders and fibres, choose one of 'systematic', 'wtmn'
    
    plot : type - boolean (defulat - True)
        whether to fit a velocity curve, if True then yes, if no then False
    
    flagged_points : type - list (default - [])
        True/ False array as to whether to keep the point in the fitting
        
    binary : type - boolean (default - False)
        Add in linear component to the velocities to account for binary motion
    
    order_remove : type - list
        Date, observation and the orders that need to be removed to fix a flagged point    
    
    Returns
    -------
    If plot = False
        mean : type - float 
            mean velocity for all points
        
        rms : type - float 
            RMS velocity for all points
    
    If plot = True
        m : type - float 
            Mass of the planet in Earth Masses
        
        m_err : type - float
            Error on mass of planet in Earth Masses
        
        flagged_points : type - list
            Boolean array of flagged points to check if want to include in the final velocity fit
        
        flagged_files : type - list 
            List of flagged points to check velocities of 
        
        phase : type - numpy nd-array
            Phase of each velocity point
        
        velocity : type - numpy nd-array
            velocity for each observation
        
        velocity_err : type - numpy nd-array
            Velocity error for each observation
        
        dates : type - numpy nd-array
            MJD for each observation
                    
        mnrvs_l : type - numpy nd-array 
            Mean velocities for each day of observations
        
        mnrvs_errs_l : type - numpy nd-array 
            Errors for the mean velocities for each day of observations
        
        dd : type - numpy nd-array
            MJD for mean velocities for each day of observations
                           
    
    """
    # if want to plot and find the mass, then star parameters should be in the known_parameters.csv file, so read them in
    if plot:
        parameters = Table.read('known_parameters.csv')
        star_parameters = parameters[parameters['\ufeffname']==star_name][0]
        star_mass = star_parameters['star_mass']
        star_mass_error = star_parameters['star_mass_error']
        period = star_parameters['period']
        period_error = star_parameters['period_error']
        epoch = star_parameters['epoch']
        epoch_error = star_parameters['epoch_error']
        inclination = star_parameters['inclination']
        inclination_error = star_parameters['inclination_error']
    if plot:
        flagged_points, flagged_files, inds, phase, velocity, velocity_err, files, day_rvs = flag_rvs(star_name, combination = combination, plot = plot, flagged_points = flagged_points, order_remove = order_remove)
        phase = phase[~np.array(flagged_points)]
        velocity = velocity[~np.array(flagged_points)]
        velocity_err = velocity_err[~np.array(flagged_points)]
    else:
        flagged_points,inds, phase, velocity, velocity_err, files, day_rvs = flag_rvs(star_name, combination = combination, plot = plot, flagged_points = flagged_points, order_remove = order_remove)
           
    if True:
        mnrvs = []
        dates = []
        mnrvs_errs = []
        mnrvs_l = []
        mnrvs_errs_l = []
        dd = []
        
        for elem in day_rvs:

            dates.extend([elem[i][0] for i in range(len(elem))])
            dd.append(np.nanmean([elem[i][0] for i in range(len(elem))]))
            vels = [elem[i][1]*1000 for i in range(len(elem))]
            vels = np.array(vels)
            velerrs = [elem[i][2]*1000 for i in range(len(elem))]
            velerrs = np.array(velerrs)
            
            # for each velocity, check if it is still in the velocity list (if it has been flagged, then don't include it in the mean day velocity)
            for ind, vel in enumerate(vels):
                k = np.where(abs(velocity - vel) < 1e-5)[0]
               
                if len(k) == 0:
                    velerrs[ind] = 1e5  
            w = 1/velerrs**2
            for i, weight in enumerate(w):
                if np.isinf(abs(weight)):
                            weights[i] = 0
            if np.nansum(abs(w))==0:
                mnrvs.extend([np.nan]*len(elem))
                mnrvs_errs.extend([np.nan]*len(elem))
                mnrvs_l.append(np.nan)
                mnrvs_errs_l.append(np.nan)
            else:
                mnrvs.extend([np.nansum(w*vels)/np.nansum(w)]*len(elem))
                mnrvs_errs.extend([1/np.nansum(w)**0.5]*len(elem))
                mnrvs_l.append(np.nansum(w*vels)/np.nansum(w))
                mnrvs_errs_l.append(1/np.nansum(w)**0.5)
                
        
        
        mnrvs = np.array(mnrvs)
        mnrvs_errs = np.array(mnrvs_errs)
        dates = np.array(dates)
            
            
        if len(velocity)!=len(mnrvs) and len(flagged_points) != 0:  
            mnrvs = mnrvs[~np.array(flagged_points)]
            mnrvs_errs = mnrvs_errs[~np.array(flagged_points)] 
            d = dates[~np.array(flagged_points)]
        else:
            d = dates
        
        a = optimise.least_squares(linear_func, x0 = [0,0], args = (d,mnrvs,mnrvs_errs))
            
        # sort dates to be in the same order as the phase plot
        dates = dates[inds]
        dates = dates[~np.array(flagged_points)]
        if binary:
            velocity += a.x[0]*dates
        
        #plt.figure()
        #plt.errorbar(dates,velocity,yerr = velocity_err,fmt='ko')
        #plt.xlabel('MJD')
        #plt.ylabel("Velocity (m/s)")
        #plt.title(star_name)
        #plt.show()
        
    if plot:                
        init_cond = [(np.nanmax(velocity)-np.nanmin(velocity))/2,np.nanmedian(velocity)]
        print(init_cond)
        
        try:    
            a = optimise.least_squares(func,x0 = init_cond, args=(np.array(phase),np.array(velocity),np.array(velocity_err),period,epoch))
        except ValueError:
            velocity_err = np.where(np.isinf(abs(velocity_err)),1e5,velocity_err)
            velocity_err = np.where(np.isnan(velocity_err),1e5,velocity_err)
            velocity = np.where(np.isnan(velocity),0,velocity)
            
            plt.plot(velocity,'.')
            plt.show()
            a = optimise.least_squares(func,x0 = init_cond, args=(np.array(phase),np.array(velocity),np.array(velocity_err),period,epoch))
        if a.success:
            try:
                cov = np.linalg.inv(np.dot(a.jac.T,a.jac))  
                mse = np.mean(a.fun**2) 
                v_error = np.sqrt(mse)*np.sqrt(cov[0,0])
            except:
                print('Singular Matrix')
                v_error = 0
                
        print(a.x)
        print('velocity error: ',v_error,' m/s')
        m, m_err = mass(a.x[0]*u.m/u.s,period*u.day,star_mass*u.M_sun, inclination, v_error*u.m/u.s, period_error*u.day, star_mass_error*u.M_sun, inclination_error)
        print('Mass (Earth Masses): ', m, ' +/-', m_err)
        x = np.linspace(0,1,10000)
    
    else:
        xs = phase
        ys = velocity
        yerr = velocity_err
    
    plot_points = np.where(np.array(mnrvs_errs_l)<1e4)
    
    #if not plot:
        #plt.errorbar(phase,velocity,yerr= velocity_err,fmt='ro')
        
    if plot:
        plt.figure()
        #plt.errorbar(-0.5+phase,velocity - a.x[1],yerr= velocity_err,fmt='bo')
        plt.errorbar(np.array(-0.5+((dd-epoch)%period)/period)[plot_points],np.array(mnrvs_l - a.x[1])[plot_points], yerr = np.array(mnrvs_errs_l)[plot_points], fmt = 'ro')
        plt.plot(-0.5+x, func(a.x,x,np.array(velocity),np.array(velocity_err),period,epoch,return_fit = True) - a.x[1],'k')
        plt.title(star_name)
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Phase')
        plt.show()
    
    
    # if not plotting, than looking at the mean and rms velocities
    if not plot:
        mnrvs = []
        mnrvs_err = []
        times = []
        for elem in day_rvs:
        
            dates = [elem[i][0] for i in range(len(elem))]
            
            vels = [elem[i][1]*1000 for i in range(len(elem))]
            
            velerrs = [elem[i][2]*1000 for i in range(len(elem))]
            
            vels = np.array(vels)
            velerrs = np.array(velerrs)
            t = Time(np.mean([elem[i][0] for i in range(len(elem))]), format='mjd').to_datetime()
            
            
            if True:
                times.extend([t])
                
                
                velerrs = np.where(velerrs>1e4, np.inf, velerrs)
                w = 1/velerrs**2
                for i, weight in enumerate(w):
                    if np.isinf(abs(weight)):
                                weights[i] = 0
                                vels[i] = 0
                                
                if np.nansum(abs(w))==0:
                    mnrvs.extend([np.nan])
                    mnrvs_err.extend([np.nan])
                    
                else:
                    mnrvs.extend([np.nansum(w*vels)/np.nansum(w)])
                    mnrvs_err.extend([1/np.nansum(w)**0.5])
                    
        mnrvs = np.array(mnrvs)
        mnrvs_err = np.array(mnrvs_err)
        mn = 0
        rms = 0
        count = 0
        plt.figure()
        #ys = np.where(abs(ys)<20,ys,np.nan)
        #mnrvs = np.where(abs(mnrvs)<1000,mnrvs,np.nan)
        
        for i in range(len(mnrvs)):
            if True:
                if not np.isnan(mnrvs[i]):
                    #mn += mnrvs[i] - np.nanmean(mnrvs)
                    #rms += (ys[i] - np.nanmean(ys))**2
                    rms += (mnrvs[i] - np.nanmean(mnrvs))**2
                    count += 1
                    #plt.errorbar(times[i], ys[i] - np.nanmean(ys),yerr = yerr[i],fmt = 'k.',capsize=5)
                    plt.errorbar(times[i], mnrvs[i]- np.nanmean(mnrvs), yerr = mnrvs_err[i], fmt = 'k.', capsize =5)
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Observation Date')
        plt.tight_layout()
        #plt.show()
        #print('mean',mn/count)
        print('rms',(rms/count)**0.5)
    
    if not plot:
        # do plot comparison to Chris
        t_chris, v_chris, v_err_chris = plot_chris('/home/ehold13/chris_'+star_name+'/vccf_nightly_velocities.txt')
        
        sorting_chris = np.argsort(t_chris)
        v_chris = v_chris[sorting_chris]
        
        sorting = np.argsort(times)
        mnrvs = mnrvs[sorting]
        
        plt.figure()
        plt.plot((mnrvs- np.nanmean(mnrvs)),v_chris, 'ko')
        plt.ylabel('Cross-Correlation Velocity (m/s)')
        plt.xlabel('Least-Squares Velocity (m/s)')
        plt.show()
    if not plot:
        return (rms/count)**0.5
    else:
        return m,m_err, flagged_points, flagged_files, phase, velocity, velocity_err, dates, mnrvs_l, mnrvs_errs_l,dd
        
        
def plot_chris(nightly_vels_dir):
    jds = []
    v_nightly = []
    v_nightly_err = []
    f = open(nightly_vels_dir,'r')
    f_lines = f.readlines()
    
    for i,f_line in enumerate(f_lines[1:]):
        
        if i not in [1,2,34,35,48,49,50,51,52,53] and nightly_vels_dir == '/home/ehold13/chris_10700/vccf_nightly_velocities.txt':
            
            jds.append(float(f_line.split('  ')[1])+0.5)
            v_nightly.append(-float(f_line.split('  ')[2]))
            v_nightly_err.append(float(f_line.split('  ')[3]))
        elif i not in [0,1,6,7,11,12,13,19,22,28,30,31,32,33,34,35,37,40,47] and i not in list(range(48,71)) and nightly_vels_dir == '/home/ehold13/chris_85512/vccf_nightly_velocities.txt':
            jds.append(float(f_line.split('  ')[1])+0.5)
            v_nightly.append(-float(f_line.split('  ')[2]))
            v_nightly_err.append(float(f_line.split('  ')[3]))
    f.close()
    
    mean = np.nanmean(v_nightly)
    rms = (np.sum((np.array(v_nightly)-mean)**2)/len(v_nightly))**0.5
    print('chris rms: ',rms)
    plt.figure()
    plt.errorbar(Time(jds, format = 'jd').to_datetime(),np.array(v_nightly) - mean,yerr = v_nightly_err,fmt = 'k.', capsize = 5)
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Observation Date')
    #plt.show()
        
    return Time(jds, format = 'jd').to_datetime(), np.array(v_nightly) - mean, v_nightly_err       
        

def chris_all(all_vels_dir):
    jds = []
    v = []
    v_err = []
    f = open(all_vels_dir,'r')
    f_lines = f.readlines()
    
    for i,f_line in enumerate(f_lines[1:]):
        #print(i,f_line.split('  '))    
        jds.append(float(f_line.split('  ')[1])+0.5)
        v.append(float(f_line.split('  ')[2]))
        v_err.append(float(f_line.split('  ')[3]))
        
    f.close()
    
    mean = np.nanmean(v)
    rms = (np.sum((np.array(v)-mean)**2)/(len(v)))**0.5
    
    print('chris rms: ', rms)
     
                         
