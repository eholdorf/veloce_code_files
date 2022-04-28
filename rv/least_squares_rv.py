import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.table import Table
from .main_funcs import log_scale_interpolation
from .main_funcs import telluric_correction
from . import get_observations
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as optimise
import astropy.constants as c
import astropy.units as u
from barycorrpy import get_BC_vel
from . import utils
from rv.main_funcs import barycentric_correction

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
    
    # find the telluric spectrum for before and after the observation
    B_plus_saved = None
    wave_tell_b, telluric_spec_b, telluric_err_spec_b, target_info_b, telluric_info_b, B_plus_saved = telluric_correction(obs_fits,'before', fits_path[41:47] ,scrunch = True, B_plus = B_plus_saved)
    
    wave_tell_a, telluric_spec_a, telluric_err_spec_a, target_info_a, telluric_info_a, B_plus_saved = telluric_correction(obs_fits,'after', fits_path[41:47], scrunch = True, B_plus = B_plus_saved)
    
    
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
            
            mask = np.isnan(all_s_logflux[:,order,fibre])
            scale = np.median(all_s_logflux[:,order,fibre][~mask])
            all_s_logflux[:,order,fibre] /= scale
            all_s_logerrflux[:,order,fibre] /= scale
    
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
        
    primary_hdu = pyfits.PrimaryHDU(spect)
    image_hdu = pyfits.ImageHDU(wavelength)
    image_hdu2 = pyfits.ImageHDU(spect_err)
    hdul = pyfits.HDUList([primary_hdu, image_hdu, image_hdu2])
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

def generate_rvs(star_name, date, template, int_guess = -20):
    
    veloce_obs = Table.read('/home/ehold13/veloce_scripts/veloce_observations.fits')

    # limit to the given star name and date
    stars = veloce_obs[veloce_obs['star_names']==star_name]

    obs_index = []

    for j,star in enumerate(stars):
        for index,yymmdd in enumerate(star[8]):
            if yymmdd.decode('utf-8') == date:
                obs_index.append((j,index))

    files = [stars[obs[0]][7][obs[1]].decode('utf-8') for obs in obs_index]

    orders = [6,7,13,14,17,25,26,27,28,30,31,33,34,35,36,37]
    rvs = np.empty((len(files),len(orders),19))
    rv_errs = np.empty((len(files),len(orders),19))
    mses = np.empty((len(files),len(orders),19))

    wtmn_rv = np.empty((len(files),len(orders)))
    wtmn_rv_err = np.empty((len(files),len(orders)))
    velocity_errors = np.empty((len(files),len(orders)))

    for fit_index,fits in enumerate(files):
        obs_file_path = '/priv/avatar/ehold13/obs_corrected/'+fits[0:10]+'_corrected.fits'
        observation = pyfits.open(obs_file_path)
        
        BC_t, BC_star = barycentric_correction('11dec30096o.fits',obs_file_path[35:45]+'o.fits','191211',date)
        if BC_star > 0:
            int_guess  = 20
        
        print('Fitting Observation '+fits+', '+str(int(fit_index)+1)+'/'+str(len(files)))
        spectrum = observation[0].data
        wavelength = observation[1].data
        error = observation[2].data
        
        for order_index,order in enumerate(orders): #enumerate(range(np.shape(spectrum)[1])):
            temp_wave = template[1].data[:,order]
            temp_spec = template[0].data[:,order]
            gaus = np.exp(-np.linspace(-2,2,15)**2/2) 
            gaus /= np.sum(gaus)
            temp_spec = np.convolve(temp_spec,gaus, mode='same')
            
            temp_func = InterpolatedUnivariateSpline(temp_wave, temp_spec, k=1) 
            temp_lwave = np.log(temp_wave)
            
            temp_dlwave = temp_lwave[1]-temp_lwave[0]
            
            
            for fibre_index,fibre in enumerate(range(np.shape(spectrum)[2])):
             
                spect = spectrum[830:3200,order,fibre]
                mask = np.isnan(spect)
                spect = spect[~mask]
                wave = wavelength[830:3200,order,fibre][~mask].astype(np.float64)
                log_wave = np.log(wave)
                err = error[830:3200,order,fibre][~mask]
                
                if 0==len(log_wave):
                    continue
                    
                a = optimise.least_squares(rv_fitting_eqn,x0 = [int_guess,0,0,0], args=(log_wave, spect, err, temp_spec, temp_lwave[0], temp_dlwave), \
                    jac=rv_jac, method='lm')    
                cov = np.linalg.inv(np.dot(a.jac.T,a.jac))    
                if a.success: 
                    rvs[fit_index,order_index,fibre_index] = a.x[0]
                    mse = np.mean(a.fun**2) 
                    mses[fit_index,order_index,fibre_index] = mse
                    rv_errs[fit_index,order_index,fibre_index] = np.sqrt(mse)*np.sqrt(cov[0,0])
                    
                
            weights = 1/rv_errs[fit_index,order_index,:]**2
            wtmn_rv[fit_index,order_index] = np.sum(weights*rvs[fit_index,order_index,:])/np.sum(weights)
            wtmn_rv_err[fit_index,order_index] = 1/np.sqrt(np.sum(weights))
                
            #obs_day = date #obs_file_path[35:37]
            #BC_t, BC_star = barycentric_correction('11dec30096o.fits',obs_file_path[35:45]+'o.fits','191211',obs_day)
            print(BC_star*c.c, wtmn_rv[fit_index,order_index]*1000)
            velocity_errors[fit_index, order_index] = (wtmn_rv[fit_index,order_index]*1000 - BC_star*c.c.to(u.m/u.s).value)
    print('Velocity uncertainty, list of orders (m/s): {:.1f}'.format(np.std(np.mean(velocity_errors, axis=1)))) # was 24:34
    print('Internal dispersion, based on scatter between orders (m/s): ')
    simple_std = np.std(velocity_errors, axis=1)/np.sqrt(len(orders))
    print(simple_std)        
    return velocity_errors, files, orders
        

