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


Tau_Ceti_Template = pyfits.open('/home/ehold13/veloce_scripts/Tau_Ceti_Template_dec2019_tellcor_1.fits')
wavelength = Tau_Ceti_Template[1].data

#print(wavelength[11300,13])
#obs = pyfits.open('/priv/avatar/velocedata/Data/spec_211202/191214/14dec30068oi_extf.fits')

#all_log_w, all_t_logflux, all_t_logerrflux, all_s_logflux, all_s_logerrflux, airmasses = log_scale_interpolation('11dec30096o.fits','14dec30068o.fits',BC=False)

#plt.figure()
#plt.plot(wavelength - all_log_w)
#plt.show()

#wave_tell_b, telluric_spec_b, telluric_err_spec_b, target_info_b, telluric_info_b = telluric_correction('11dec30096o.fits','14dec30068o.fits','before')
#wave_tell_a, telluric_spec_a, telluric_err_spec_a, target_info_a, telluric_info_a = telluric_correction('11dec30096o.fits','14dec30068o.fits','after')

#if telluric_info_a[3]!= telluric_info_b[3]:
#    telluric_spec = (telluric_spec_a*(target_info_b[3] - telluric_info_b[3]) + telluric_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3])
#else:
#    telluric_spec = telluric_spec_a

#for fibre in range(19):    
#    all_s_logflux[:,:,fibre] /= telluric_spec
    
#spect = all_s_logflux

#spect_err = all_s_logerrflux

#spect = np.zeros((22600,40))
#spect_err = np.zeros((22600,40))
#for fibre in range(19):
#    for order in range(40):
        
#        scale = np.median(all_s_logflux[:,order,fibre])
#        spect[:,order] += all_s_logflux[:,order,fibre]/scale
#        spect_err[:,order] += all_s_logerrflux[:,order,fibre]/scale
        
#primary_hdu = pyfits.PrimaryHDU(spect)
#image_hdu = pyfits.ImageHDU(all_log_w)
#image_hdu2 = pyfits.ImageHDU(spect_err)
#hdul = pyfits.HDUList([primary_hdu, image_hdu, image_hdu2])
#hdul.writeto('14dec30068_corrected.fits')

obs = pyfits.open('/home/ehold13/veloce_scripts/rv/14dec30068_corrected.fits')

spect = obs[0].data
all_log_w = obs[1].data
spect_err = obs[2].data

#plt.figure()
#plt.plot(wavelength - all_log_w)
#plt.show()

 
#temp_func = InterpolatedUnivariateSpline(Tau_Ceti_Template[1].data[:,:],Tau_Ceti_Template[0].data[:,:],k=5)

#plt.figure()
#plt.plot(Tau_Ceti_Template[1].data[:,13],19*Tau_Ceti_Template[0].data[:,13])

#plt.figure()
#plt.plot(all_log_w[:,13],spect[:,13])

#plt.show()

def rv_fitting_eqn(params, wave, spect, spect_err, interp_func):
    delta_wave = [wave[i+1] - wave[i] for i in range(len(wave)-1)]
    delta_wave.append(delta_wave[-1])
    delta_wave = np.array(delta_wave)
    
    delta_wave = 1.08 * np.mean([wave[0],wave[-1]])
    
    scaling_factor = np.exp(params[1]+params[2]*delta_wave + params[3]*delta_wave**2)
    
    fitted_spectra = interp_func(wave * (1.0 + params[0]/c.c.si.value))*scaling_factor
    return (fitted_spectra - spect)/spect_err
    
val = []
for j in range(1):
    for i in range(40):  
        temp_func = InterpolatedUnivariateSpline(Tau_Ceti_Template[1].data[:,i],Tau_Ceti_Template[0].data[:,i],k=1) 
        bad = np.zeros(len(spect))
 
        spect_,bplus = utils.correct_bad(spect[:,i,j],bad)   
        a = optimise.leastsq(rv_fitting_eqn,x0 = [-24000,0,0,0],
        args=(all_log_w[
    :,i], spect_, spect_err[:,i,j], temp_func),full_output = True)
        print(a[0])
    #plt.plot(all_log_w[:,i],a[2]['fvec'])
    #plt.title(str(i))
    #plt.show()
        err = [a[2]['fvec'][i]**2 for i in range(len(a[2]['fvec']))]
        print(np.sqrt(sum(err)))
        if a[0][0] != -24000:
            val.append((a[0][0],np.sqrt(sum(err))))

good_val = []

for elem in val:
    if elem[0] != -24000:
        good_val.append(elem[0])
        
print(np.median(good_val))
    
rv = 0
weights = 0
for elem in val:
    if elem[0] != -24000:
        weight = (1/elem[1])**2
        rv += weight * elem[0]
    
        weights += weight
    
rv /= weights
rv_err = 1/weights**0.5
print(rv,rv_err)

BC_t, BC_star = barycentric_correction('11dec30096o.fits','14dec30068o.fits')

print(100*(np.median(good_val) - BC_star*c.c.si.value)/(BC_star*c.c.si.value))

#obs = pyfits.open('/home/ehold13/veloce_scripts/rv/14dec30068_corrected.fits')

#spect = obs[0].data
#all_log_w = obs[1].data
#all_log_w_ = all_log_w + 3/c.c.si.value*all_log_w
#spect_err = obs[2].data

#val = []
#for j in range(1):
#    for i in range(40):  
#        temp_func = InterpolatedUnivariateSpline(all_log_w[:,i],spect[:,i,j],k=3)    
#        a = optimise.leastsq(rv_fitting_eqn,x0 = [-1,0,0,0],
#        args=(all_log_w_[
#    :,i], spect[:,i,j], spect_err[:,i,j], temp_func),full_output = True)
#        print(a[0])
    #plt.plot(all_log_w[:,i],a[2]['fvec'])
    #plt.title(str(i))
    #plt.show()
#        err = [a[2]['fvec'][i]**2 for i in range(len(a[2]['fvec']))]
#        print(np.sqrt(sum(err)))
#        val.append((a[0][0],np.sqrt(sum(err))))
#good_vals = []
#for elem in val:
#    if elem[0] != -1:
#        good_vals.append(elem)
#print(np.median(good_vals),0)

#rv = 0
#weights = 0
#for elem in good_vals: 
#    weight = 1/elem[1]**2
#    rv += elem[0]*weight
#    weights += weight
    
#rv /= weights
#print(rv)

#print(100*(np.median(good_vals) - 3)/(3))
    
    
    
    
