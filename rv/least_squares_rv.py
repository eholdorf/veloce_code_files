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


def create_observation_fits(standard, obs_fits, date, save_dir, combine_fibres = False):
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
    

def combination_method_two(observation_dir = '/home/ehold13/veloce_scripts/veloce_reduction/10700/', dispersion_limit = 0.1):
    all_obs_rvs = []
    all_order_rvs = []
    
    # iterate over the Tau Ceti Observations
    for fit_index,fits in enumerate(os.listdir(observation_dir)):
        if fits.endswith('.fits'):  # fits in ['fits_191211.fits']:  #fits.endswith('.fits'):
            observations = pyfits.open(observation_dir + fits)
            
            # now take weighted mean over the fibres to get a velocity per order for each observation
            order_rv = np.empty((len(observations['RV'].data[:,0,0]),len(observations['RV'].data[0,:,0])))
            
            order_rv_err = np.empty((len(observations['RV'].data[:,0,0]),len(observations['RV'].data[0,:,0])))
            
            fit_rv = np.empty(len(observations['RV'].data[:,0,0]))
            fit_rv_err = np.empty(len(observations['RV'].data[:,0,0]))
                    
            # for each observation on this date check to see if have low dispersion (i.e. is a good observation)
            for obs in range(len(observations['RV'].data[:,0,0])):
                if np.std(observations['RV'].data[obs,3:,:]) < dispersion_limit:
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
                            if np.nansum(weights) == 0:
                                order_rv[obs,order] = 0
                                order_rv_err[obs,order] = 0
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

                               
            all_obs_rvs.extend(fit_rv)
            all_order_rvs.extend(order_rv)

    all_obs_rvs = np.array(all_obs_rvs)
    all_order_rvs = np.array(all_order_rvs)
    
    all_obs_rvs = np.where(np.isinf(all_obs_rvs),np.nan,all_obs_rvs)
    
    all_order_rvs= np.where(np.isinf(all_order_rvs),np.nan,all_order_rvs)
    
    mean_sq_resid = np.nanmedian((all_obs_rvs - np.transpose(all_order_rvs))**2,1)
    
     
    return mean_sq_resid
            
    
def combination_method_three(observation_dir, dispersion_limit = 0.1):
    median_flux = []
    v_wtmn = []
    for fits in os.listdir(observation_dir):
        if fits.endswith('.fits'):
            observations = pyfits.open(observation_dir + fits)
            for obs in range(len(observations['RV'].data[:,0,0])):
                if np.std(observations['RV'].data[obs,3:,:]) < dispersion_limit:
                    median_flux.append(observations['median_flux'].data[obs,:,:])
                    v_wtmn.append(observations['RV'].data[obs,:,:])

    median_flux = np.array(median_flux)
    v_wtmn = np.array(v_wtmn)
    order_means = np.mean(np.mean(median_flux,0),1)
    good_orders = np.where(order_means == order_means)[0]

    median_flux = median_flux[:,good_orders,:]
    v_wtmn = v_wtmn[:,good_orders,:]
    
    for obs in range(np.shape(median_flux[:,0,0])[0]):
        for order in range(np.shape(median_flux[0,:,0])[0]):
            median_flux[obs,order,:] /= np.mean(median_flux[obs,order,:])
            v_wtmn -= np.mean(v_wtmn[obs,order,:])
            
    X = median_flux.reshape((median_flux.shape[0],median_flux.shape[1]*median_flux.shape[2]))
    v_wtmn = v_wtmn.reshape((v_wtmn.shape[0],v_wtmn.shape[1]*v_wtmn.shape[2]))             
    
    W,V = np.linalg.eigh(np.dot(X.T,X))
    
    A = V[:,-4:]
    
    Y = np.dot(X,A)
    p = np.dot(np.dot(np.linalg.inv(np.dot(Y.T,Y)),Y.T),v_wtmn)
    
    v_adjust = v_wtmn - np.dot(Y,p)
    
    plt.imshow(v_adjust, aspect = 'auto', interpolation = 'nearest')
    plt.show()
    
    plt.imshow(p*1000, aspect = 'auto', interpolation = 'nearest')
    plt.colorbar()
    plt.show()
              

def generate_rvs(star_name, date, template_path, int_guess = 0.1, alpha = 0.2, residual_limit = 0.5,runs = 1, total_runs = 5):   
    
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
                        initial_cond = [int_guess,0,0,0]
                        a = optimise.least_squares(rv_fitting_eqn,x0 = initial_cond, args=(log_wave, spect, err, temp_spec, temp_lwave[0], temp_dlwave), \
                            jac=rv_jac, method='lm')
                        
                        for i,value in enumerate(a.fun):
                            if abs(value) > residual_limit:
                                err[i] = np.inf
                        runs += 1
                    
                    if order == 130:
                        plt.figure()
                        plt.plot(wave,spect,label='original')
                        plt.plot(wave,rv_fitting_eqn(a.x,log_wave, spect, err, temp_spec, temp_lwave[0], temp_dlwave, return_spec = True),label = 'fitted')
                        plt.legend()
                        plt.show()    
                           
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
    dd = Table.read('/home/ehold13/veloce_scripts/veloce_observations.fits')
    
    stars = dd[dd['star_names']==star_name]
    for star in stars:
    
        for i, fit in enumerate(star[7]):
            if fit.decode('utf-8') != '':
                if not os.path.exists('/priv/avatar/ehold13/obs_corrected/'+star_name+'/'+fit.decode('utf-8')[0:10]+'_corrected.fits'):
                    print(fit)
                    create_observation_fits('11dec30096o.fits',fit.decode('utf-8'),star[8][i].decode('utf-8'),'/priv/avatar/ehold13/obs_corrected/'+star_name+'/')

def wtmn_combination(star_name):
    all_rvs = []
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
                    if np.nansum(weights) == 0:
                        order_rv[obs,order] = 0
                        order_rv_err[obs,order] = 0
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
                    
                dispersion_date.append((observations[4].data['MJDs'][obs],fit_rv[obs],fit_rv_err[obs]))
            
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
                all_rvs.append((np.mean(observations[4].data['MJDs']),rv,err))
    return all_rvs

def systematic_error_combination(star_name):
    all_rvs = []
    combination = combination_method_two()
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

                errors = np.where(errors<10e-16,0,errors)

                # combine fibres with weighted-mean
                for order in range(len(rvs[:,0])):
                    weights =  1/errors[order,:]**2
                    for i,weight in enumerate(weights):
                        if np.isinf(weight):
                            weights[i] = 0
                    if np.nansum(weights) == 0:
                        order_rv[obs,order] = 0
                        order_rv_err[obs,order] = 0
                    else:
                        order_rv[obs,order] = np.nansum(weights*rvs[order,:])/np.nansum(weights)
                        order_rv_err[obs,order] = 1/np.sqrt(np.nansum(weights))
                        q[order] = np.sqrt(np.nanmax([combination[order]**2 - order_rv_err[obs,order]**2,0]))   
                for i, value in enumerate(q):
                    if value > 400:
                        q[i] = np.inf
                order_rv_err[obs,:] = np.sqrt(order_rv_err[obs,:]**2 + q**2)
                 
                
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
            
                dispersion_date.append((observations[4].data['MJDs'][obs],fit_rv[obs],fit_rv_err[obs]))
            
            disp_rvs = [dispersion_date[i][1] for i in range(len(dispersion_date))]
            
            if np.nanstd(disp_rvs)<1:
                all_rvs.extend(dispersion_date)  
      
            weights = 1/fit_rv_err**2
            for i,weight in enumerate(weights):
                    if np.isinf(abs(weight)):
                        weights[i] = 0
            if np.nansum(abs(weights))==0:
                rv = np.nan
                err = np.nan
            else:
                rv = np.nansum(weights*fit_rv)/np.nansum(weights)
                err = 1/np.sqrt(np.nansum(weights))
                
            if np.nanstd(disp_rvs) < 1:     
                all_rvs.append((np.mean(observations[4].data['MJDs']),rv,err))
        
    return all_rvs

def func(params,x,y ,yerr,period,epoch,return_fit = False):
    
    if return_fit:
        return (params[0]*np.sin(2*np.pi/period*x+((epoch)%period)) +params[1])
    else:
        return (params[0]*np.sin(2*np.pi/period*x+(epoch%period)) +params[1] - y)/yerr

def mass(v,T,M_s):
    return (T/(2*np.pi*c.G))**(1/3) * abs(v) * M_s**(2/3)
               
def plot_rvs(star_name,star_mass,period,epoch, combination = 'wtmn'):
    
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
    
    
    
    if combination == 'wtmn':    
        all_rvs = wtmn_combination(star_name)
    elif combination == 'systematic':
        all_rvs = systematic_error_combination(star_name)
    
    xs1 = [((all_rvs[i][0]-epoch)%period) for i in range(len(all_rvs))]
    ys1 = [all_rvs[i][1]*1000 for i in range(len(all_rvs))]
    yerr1 = [all_rvs[i][2]*1000 for i in range(len(all_rvs))]
    
    xs1 = np.array(xs1)
    ys1 = np.array(ys1)
    yerr1 = np.array(yerr1)
    
    xs = []
    ys = []
    yerr = []
    med = np.median(ys1)
    
    ws = tk.Tk()
    ws.title('Set Upper and Lower RV Bounds')
    min_rvs = tk.IntVar(ws)
    max_rvs = tk.IntVar(ws)
    
    data1 = {'MJD': xs1,
         'RV': ys1
        }
        
    df1 = DataFrame(data1,columns=['MJD','RV'])
    figure1 = plt.Figure(figsize=(12,10), dpi=100)
    ax1 = figure1.add_subplot(111)
    ax1.scatter(df1['MJD'],df1['RV'], color = 'k')
    scatter1= FigureCanvasTkAgg(figure1, ws) 
    scatter1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    L1 = tk.Label(ws, text="Minimum")
    L1.pack()
    min_rv = tk.Entry(ws)
    min_rv.pack()
    L2 = tk.Label(ws, text="Maximum")
    L2.pack()
    max_rv = tk.Entry(ws)
    max_rv.pack()
    
    tk.Button(ws,text="Okay",command =lambda:[min_rvs.set(min_rv.get()),max_rvs.set(max_rv.get()),ws.destroy()]).pack()
    ws.mainloop()
    
    
    good_points = np.where((ys1<max_rvs.get())&(min_rvs.get()<ys1))
  
    xs1 = xs1[good_points]
    ys1 = ys1[good_points]
    yerr1 = yerr1[good_points]
    
    inds = xs1.argsort()
    xs1 = xs1[inds]
    ys1 = ys1[inds]
    yerr1 = yerr1[inds]
    
    i = 0
    rej = []
    while i < len(ys1):
        val = ys1[i]
        ws = tk.Tk()
        ws.title('Keep Point?')
        keep = tk.BooleanVar(ws)
        j = tk.IntVar(ws)  
        
        data1 = {'MJD': xs1[i:],
         'RV': ys1[i:]
        }
        
        df1 = DataFrame(data1,columns=['MJD','RV'])
    
    
        data2 = {'MJD': [xs1[i]],
         'RV': [ys1[i]]
        }
        
        df2 = DataFrame(data2,columns=['MJD','RV'])
        
        data3 = {'MJD': xs1[:i],
         'RV': ys1[:i]
        }
        
        df3 = DataFrame(data3,columns=['MJD','RV'])
        
        if len(rej)!= 0:
            data4 = {'MJD': xs1[np.array(rej)],
             'RV': ys1[np.array(rej)]
            }
            
            df4 = DataFrame(data4,columns=['MJD','RV'])

        figure1 = plt.Figure(figsize=(12,10), dpi=100)
        ax1 = figure1.add_subplot(111)
        ax1.scatter(df1['MJD'],df1['RV'], color = 'k',label = 'To Do')
        ax1.scatter(df3['MJD'],df3['RV'], color = 'g', label = 'Accepted',marker = '^')
        if len(rej)!=0:
            ax1.scatter(df4['MJD'],df4['RV'], color = 'r', marker = 'X', s = 100, label = 'Rejected')
        ax1.scatter(df2['MJD'],df2['RV'], color = 'orange', marker = '*', s = 150, label = 'Current Point')
        ax1.legend()
        scatter1= FigureCanvasTkAgg(figure1, ws) 
        scatter1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        
        b = tk.Button(ws,text = 'Keep', command =lambda:[keep.set(1),ws.destroy()]).pack()
        b = tk.Button(ws,text = 'Reject', command =lambda:[keep.set(0),ws.destroy()]).pack()
        b = tk.Button(ws,text = 'Keep Rest', command =lambda:[j.set(len(ys1)),ws.destroy()]).pack()
        ws.mainloop()
        if keep.get(): 
            xs.append(xs1[i])
            ys.append(ys1[i])
            yerr.append(yerr1[i])
        else:
            rej.append(i)
        if j.get()==len(ys1):
            xs.extend(xs1[i:])
            ys.extend(ys1[i:])
            yerr.extend(yerr1[i:])
            i = j.get()
        i += 1
    init_cond = [(max(ys)-min(ys))/2,np.nanmedian(ys)]
    print(init_cond)
    a = optimise.least_squares(func,x0 = init_cond, args=(np.array(xs),np.array(ys),np.array(yerr),period,epoch))

    print(a.x)
    print(mass(a.x[0]*u.m/u.s,period*u.day,star_mass*u.M_sun).to(u.M_earth))
    x = np.linspace(0,period,10000)
    plt.figure()
    plt.errorbar(xs,ys,yerr= yerr,fmt='ro')
    plt.plot(x, func(a.x,x,np.array(ys),np.array(yerr),period,epoch,return_fit = True),'k')
    plt.title(star_name)
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Phase')
    plt.show()
                   
            
