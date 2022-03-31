import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.table import Table
from main_funcs import log_scale_interpolation
from main_funcs import telluric_correction
from main_funcs import barycentric_correction
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as optimise
import astropy.constants as c
from barycorrpy import get_BC_vel
import utils

def create_observation_fits(standard, obs_fits, save_dir, combine_fibres = False):
    #all_log_w, all_t_logflux, all_t_logerrflux, all_s_logflux, all_s_logerrflux, airmasses = log_scale_interpolation(standard,obs_fits,BC=False)
    
    dd = pyfits.open('/priv/avatar/velocedata/Data/spec_211202/191211/'+obs_fits[0:10]+'oi_extf.fits')
    all_log_w = dd[1].data[:,:,4:23]
    all_s_logflux = dd[0].data[:,:,4:23]
    all_s_logerrflux = dd[2].data[:,:,4:23]
    plt.figure()
    plt.plot(all_log_w[:,:,0], all_s_logflux[:,:,0])
    plt.show()

    wave_tell_b, telluric_spec_b, telluric_err_spec_b, target_info_b, telluric_info_b = telluric_correction(standard,obs_fits,'before',scrunch = False)
   
    
    wave_tell_a, telluric_spec_a, telluric_err_spec_a, target_info_a, telluric_info_a = telluric_correction(standard,obs_fits,'after', scrunch = False)
   

    if telluric_info_a[1]!= telluric_info_b[1]:
        telluric_spec = (telluric_spec_a*(target_info_b[3] - telluric_info_b[3]) + telluric_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3])
    else:
        telluric_spec = telluric_spec_a
         
    
    for fibre in range(19):
        all_s_logflux[:,:,fibre] /= telluric_spec
        for order in range(40):
            mask = np.isnan(all_s_logflux[:,order,fibre])
            scale = np.median(all_s_logflux[:,order,fibre][~mask])
            all_s_logflux[:,order,fibre] /= scale
            all_s_logerrflux[:,order,fibre] /= scale
    
    wavelength = all_log_w
    spect = all_s_logflux
    spect_err = all_s_logerrflux
    
    if combine_fibres:

        spect = np.zeros((22600,40))
        spect_err = np.zeros((22600,40))
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


def rv_fitting_eqn(params, wave, spect, spect_err, interp_func, return_spec = False):
    pixel = (wave-0.5*(wave[0]+wave[-1]))/(wave[-1]-wave[0])
    
    scaling_factor = np.exp(params[1]+params[2]*pixel + params[3]*pixel**2)
    
    fitted_spectra = interp_func(wave * (1.0 + params[0]/c.c.si.value))*scaling_factor
    
    if return_spec:
        return fitted_spectra
    return (fitted_spectra - spect)/spect_err

if __name__=="__main__":

    save_dir = '/home/ehold13/veloce_scripts/obs_corrected_fits/'
    
    #s,w,se = create_observation_fits('11dec30096o.fits','11dec30096o.fits',save_dir)
    #s,w,se = create_observation_fits('11dec30096o.fits','14dec30068o.fits',save_dir)
    
    Tau_Ceti_Template = pyfits.open('/home/ehold13/veloce_scripts/Tau_Ceti_Template_dec2019_tellcor_1.fits')
    HD85512_Template = pyfits.open('/home/ehold13/veloce_scripts/HD85512_dec2019.fits')
    
    obs = pyfits.open(save_dir + '11dec30096_corrected.fits')
    spect = obs[0].data
    wavelength = obs[1].data
    spect_err = obs[2].data
    temp_func = InterpolatedUnivariateSpline(Tau_Ceti_Template[1].data[:,13], Tau_Ceti_Template[0].data[:,13], k=1)
    sp = rv_fitting_eqn([-24000,0,0,0],wavelength[401:3601,13,0],spect[401:3601,13,0],spect_err[401:3601,13,0],temp_func, return_spec = True)
    plt.figure()
    plt.plot(wavelength[401:3601,13,0],spect[401:3601,13,0])
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    plt.title('Original Spectrum')
    plt.figure()
    plt.plot(wavelength[401:3601,13,0],sp)
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    plt.title('Fitted Spectrum')
    plt.figure()
    plt.plot(wavelength[401:3601,13,0],100*(sp-spect[401:3601,13,0])/spect[401:3601,13,0])
    #plt.plot(100*(sp-spect[401:3601,13,0])/spect[401:3601,13,0])
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Percentage Error')
    plt.title('Residual')
    plt.show()
            
    #plt.figure()
    #plt.plot(wavelength[:,:,0],spect[:,:,0])
    #plt.figure()
    #plt.plot(HD85512_Template[1].data,HD85512_Template[0].data)
    #plt.figure()
    #plt.plot(Tau_Ceti_Template[1].data, Tau_Ceti_Template[0].data)
    #plt.show()
        
    val = []
    for j in range(19):
        #orders= list(range(2,15))
        #orders.extend(list(range(17,40)))
        orders = [13]
        for i in orders:
        
            #temp_mask = np.isnan(Tau_Ceti_Template[0].data[:,i])
            temp_wave = Tau_Ceti_Template[1].data[:,i]#[~temp_mask]
            temp_spec = Tau_Ceti_Template[0].data[:,i]#[~temp_mask]  
            temp_func = InterpolatedUnivariateSpline(temp_wave, temp_spec, k=1) 
            
            #spect_mask = np.isnan(spect[:,i,j])
            spect_wave = wavelength[500:1500,i,j]#[~spect_mask]
            spect_spec = spect[500:1500,i,j]#[~spect_mask]
            spect_err_ = spect_err[500:1500,i,j]#[~spect_mask]
              
            a = optimise.leastsq(rv_fitting_eqn,x0 = [-24000,0,0,0], args=(spect_wave, spect_spec, spect_err_, temp_func),full_output = True)
            
            print(a[0])
   
            plt.figure()
            plt.plot(spect_wave,a[2]['fvec'])
            plt.show()
            
            if a[0][0] != -24000:
               val.append(a[0][0])
            #a = optimise.least_squares(rv_fitting_eqn, x0 = [-24000,0,0,0], args=(spect_wave, spect_masked, spect_err_, temp_func))
            #if a.x[0] != -24000:
            #    val.append(a.x[0])
            #print(a.x)

    print(np.median(val))

    BC_t, BC_star = barycentric_correction('11dec30096o.fits','14dec30068o.fits')
    
    print(BC_t*c.c)

    print(100*(np.median(val) - BC_t*c.c.si.value)/(BC_t*c.c.si.value))
    
    own = False
    if own:
        obs = pyfits.open('/home/ehold13/veloce_scripts/obs_corrected_fits/14dec30068_corrected.fits')
        spect = obs[0].data
        all_log_w = obs[1].data
        all_log_w_ = all_log_w + 3/c.c.si.value*all_log_w
        spect_err = obs[2].data

        val = []
        for j in range(19):
            for i in range(40):  
                temp_func = InterpolatedUnivariateSpline(all_log_w[:,i],spect[:,i,j],k=1)    
                a = optimise.leastsq(rv_fitting_eqn,x0 = [-1,0,0,0],
                args=(all_log_w_[
            :,i], spect[:,i,j], spect_err[:,i,j], temp_func),full_output = True)
                print(a[0])
                
                if a[0][0] != -1:
                    val.append(a[0][0])
        print(np.median(val))

        print(100*(np.median(val) + 3)/(3))
    
    
    
    
