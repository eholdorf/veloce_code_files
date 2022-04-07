import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.table import Table
<<<<<<< HEAD
from main_funcs import telluric_correction
from main_funcs import barycentric_correction
=======
from .main_funcs import log_scale_interpolation
from .main_funcs import telluric_correction
from .main_funcs import barycentric_correction
>>>>>>> b4eaaf793b1dc113b3a5c534f6025f915effa11e
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as optimise
import astropy.constants as c
import astropy.units as u
from barycorrpy import get_BC_vel
from . import utils
from . import get_observations

def create_observation_fits(standard, obs_fits, date, save_dir, combine_fibres = False):
    file_path = get_observations.get_fits_path([obs_fits.encode('utf-8')])
    for path in file_path:
        if path[41:47] == date:
            fits_path = path
       
    dd = pyfits.open(fits_path)
    all_log_w = dd[1].data[:,:,4:23]
    all_s_logflux = dd[0].data[:,:,4:23]
    all_s_logerrflux = dd[2].data[:,:,4:23]

    B_plus_saved = None
    wave_tell_b, telluric_spec_b, telluric_err_spec_b, target_info_b, telluric_info_b, B_plus_saved = telluric_correction(obs_fits,'before', fits_path[41:47] ,scrunch = True, B_plus = B_plus_saved)
    
    wave_tell_a, telluric_spec_a, telluric_err_spec_a, target_info_a, telluric_info_a, B_plus_saved = telluric_correction(obs_fits,'after', fits_path[41:47], scrunch = True, B_plus = B_plus_saved)
    
  
    if telluric_info_a[1]!= telluric_info_b[1]:
        telluric_spec = (telluric_spec_a*(target_info_b[3] - telluric_info_b[3]) + telluric_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3])
        telluric_err_spec = (((telluric_spec_a*(target_info_b[3] - telluric_info_b[3]))/(telluric_info_a[3]-telluric_info_b[3]))**2 + ((telluric_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3]))**2)**0.5
        
    else:
        telluric_spec = telluric_spec_a
        telluric_err_spec = telluric_err_spec_a    
    
    telluric = np.zeros([np.shape(all_log_w)[0],np.shape(all_log_w)[1]])
    telluric_error = np.zeros([np.shape(all_log_w)[0],np.shape(all_log_w)[1]])
    for fibre in range(19):
        for order in range(40):        
            telluric_interpolation_func = InterpolatedUnivariateSpline(wave_tell_a[:,order], telluric_spec[:,order],k=5)
            telluric_err_interpolation_func = InterpolatedUnivariateSpline(wave_tell_a[:,order], telluric_err_spec[:,order],k=5)
            telluric[:,order] = telluric_interpolation_func(all_log_w[:,order,fibre])
            telluric_error[:,order] = telluric_err_interpolation_func(all_log_w[:,order,fibre])
            for wave in range(np.shape(all_log_w)[0]):
                all_s_logerrflux[wave,order,fibre] = ((all_s_logerrflux[wave,order,fibre]/all_s_logflux[wave,order,fibre])**2 + (telluric_error[wave,order]/telluric[wave,order])**2)**0.5
            all_s_logflux[:,order,fibre] /= telluric[:,order]
        
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
    relativistic factor (~v/c) if deriv=True
    """
    ix = (lwave - lwave0)/dlwave
    #Don't go past the edges.
    ix_int = np.maximum(np.minimum(ix.astype(int), len(template)-2), 0)
    frac = np.maximum(np.minimum(ix - ix_int, 1), 0)
    if deriv:
        #The derivative of the return line below with respect to frac, divided by dlwave
        return (template[ix_int+1] - template[ix_int])/dlwave
    else:
        return template[ix_int]*(1-frac) + template[ix_int+1]*frac


def rv_jac(params, wave, spect, spect_err, interp_func,vo = 0, ve = 0):
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


