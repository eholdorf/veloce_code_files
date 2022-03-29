import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.table import Table
from main_funcs import log_scale_interpolation
from main_funcs import telluric_correction
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as optimise
import astropy.constants as c
from barycorrpy import get_BC_vel


Tau_Ceti_Template = pyfits.open('/home/ehold13/veloce_scripts/Tau_Ceti_Template_dec2019_tellcor_1.fits')
wavelength = Tau_Ceti_Template[1].data

#print(wavelength[11300,13])
#obs = pyfits.open('/priv/avatar/velocedata/Data/spec_211202/191214/14dec30068oi_extf.fits')

all_log_w, all_t_logflux, all_t_logerrflux, all_s_logflux, all_s_logerrflux, airmasses = log_scale_interpolation('11dec30096o.fits','11dec30096o.fits',BC=False)

wave_tell_b, telluric_spec_b, telluric_err_spec_b, target_info_b, telluric_info_b = telluric_correction('11dec30096o.fits','11dec30096o.fits','before')
wave_tell_a, telluric_spec_a, telluric_err_spec_a, target_info_a, telluric_info_a = telluric_correction('11dec30096o.fits','11dec30096o.fits','after')

if telluric_info_a[3]!= telluric_info_b[3]:
    telluric_spec = (telluric_spec_a*(target_info_b[3] - telluric_info_b[3]) + telluric_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3])
else:
    telluric_spec = telluric_spec_a

for fibre in range(19):    
    all_s_logflux[:,:,fibre] /= telluric_spec
    
spect = all_s_logflux

spect_err = all_s_logerrflux

spect = np.zeros((22600,40))
spect_err = np.zeros((22600,40))
for fibre in range(19):
    for order in range(40):
        
        scale = np.median(all_s_logflux[:,order,fibre])
        spect[:,order] += all_s_logflux[:,order,fibre]/scale
        spect_err[:,order] += all_s_logerrflux[:,order,fibre]/scale
        
#primary_hdu = pyfits.PrimaryHDU(spect)
#image_hdu = pyfits.ImageHDU(all_log_w)
#image_hdu2 = pyfits.ImageHDU(spect_err)
#hdul = pyfits.HDUList([primary_hdu, image_hdu, image_hdu2])
#hdul.writeto('11dec30096_corrected.fits')

#obs = pyfits.open('/home/ehold13/veloce_scripts/11dec30096_corrected.fits')

#spect = obs[0].data
#wave = obs[1].data
#spect_err = obs[2].data
 
#temp_func = InterpolatedUnivariateSpline(Tau_Ceti_Template[1].data[:,:],Tau_Ceti_Template[0].data[:,:],k=5)

#plt.figure()
#plt.plot(Tau_Ceti_Template[1].data[:,13],19*Tau_Ceti_Template[0].data[:,13])

#plt.figure()
#plt.plot(all_log_w[:,13],spect[:,13])

#plt.show()

def rv_fitting_eqn(params, wave, spect, spect_err, interp_func):
    delta_wave = abs(wave[0] - wave[0])*np.arange(len(wave))
    scaling_factor = np.exp(params[1]+params[2]*delta_wave + params[3]*delta_wave**2)
    
    fitted_spectra = interp_func(wave * (1.0 + params[0]/c.c.si.value))*scaling_factor
    return (fitted_spectra - spect)/spect_err
    
val = []
for i in range(1):  
    temp_func = InterpolatedUnivariateSpline(Tau_Ceti_Template[1].data[3000:4300,13],Tau_Ceti_Template[0].data[3000:4300,13],k=3)    
    a = optimise.leastsq(rv_fitting_eqn,x0 = [-24000,0,0,0],
    args=(all_log_w[
    3000:4300,13], spect[3000:4300,13], spect_err[3000:4300,13], temp_func),full_output = True)
    print(a[0])
    plt.plot(wavelength[3000:4300,13],a[2]['fvec'])
    plt.title(str(i))
    plt.show()
    val.append(a[0][0])
print(np.mean(val))
print(np.median(val))
    
    
    
    
