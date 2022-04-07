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
    

if __name__=="__main__":

    save_dir = './' #'/home/ehold13/veloce_scripts/obs_corrected_fits/'
    #files = ['11dec30097o.fits', '12dec30132o.fits', '12dec30133o.fits', '12dec30134o.fits', '13dec30076o.fits', '13dec30077o.fits', '14dec30066o.fits', '14dec30067o.fits', '15dec30097o.fits', '15dec30098o.fits', '15dec30099o.fits']
    #for file_name in files:
        #s,w,se = create_observation_fits('11dec30096o.fits',file_name,'/home/ehold13/veloce_scripts/obs_corrected_fits/')
    #s,w,se = create_observation_fits('11dec30096o.fits','14dec30068o.fits','/home/ehold13/veloce_scripts/obs_corrected_fits/')
    fitting_files = ['11dec30096','11dec30097','12dec30132','12dec30133','12dec30134', '13dec30076','13dec30077','14dec30066','14dec30067','14dec30068','15dec30097', '15dec30098', '15dec30099']
    fitting_dates = ['191211','191211','191212','191212','191212','191213','191213','191214','191214','191214','191215','191215','191215']
    velocity_err = np.zeros([len(fitting_files),36])
    file_ind = 0
    for fit_index,fit in enumerate(fitting_files):
        Tau_Ceti_Template = pyfits.open('/home/ehold13/veloce_scripts/Tau_Ceti_Template_14dec2019_telluric_patched.fits')
        
        obs_file_path = '/home/ehold13/veloce_scripts/obs_corrected_fits/'+fit+'_corrected.fits'
        obs = pyfits.open(obs_file_path)

        spect = obs[0].data
        wavelength = obs[1].data
        spect_err = obs[2].data
        
            
        if False:
            order = 13
              
            temp_func = InterpolatedUnivariateSpline(Tau_Ceti_Template[1].data[:,order], Tau_Ceti_Template[0].data[:,order], k=1)
            sp = rv_fitting_eqn([-24,1e-3,1e-1,1e-1],wavelength[401:3601,order,0],spect[401:3601,order,0],spect_err[401:3601,order,0],temp_func, return_spec = True)

            plt.figure()
            plt.plot(wavelength[401:3601,order,0],spect[401:3601,order,0], label = 'Original Spectrum')
            plt.plot(wavelength[401:3601,order,0],sp, label = 'Fitted Spectrum')
            plt.xlabel('Wavelength ($\AA$)')
            plt.ylabel('Flux')
            plt.title(fit)
            plt.legend(loc='best')
            plt.show()
            

        orders= list(range(2,15))
        orders.extend(list(range(17,40)))
        order_ind = 0
        #orders = [13]
        for i in orders:
            temp_mask = np.isnan(Tau_Ceti_Template[0].data[:,i])
            temp_wave = Tau_Ceti_Template[1].data[:,i][~temp_mask]
            temp_spec = Tau_Ceti_Template[0].data[:,i][~temp_mask]  
            temp_func = InterpolatedUnivariateSpline(temp_wave, temp_spec, k=1) 
            rvs = []
            rv_errs = []
            for j in range(19):
                spect_mask = np.isnan(spect[401:3601,i,j])
                spect_wave = wavelength[401:3601,i,j][~spect_mask]
                spect_spec = spect[401:3601,i,j][~spect_mask]
                spect_err_ = spect_err[401:3601,i,j][~spect_mask]
                
                # changing intial conditions changes answer and MSE 
                a = optimise.leastsq(rv_fitting_eqn,x0 = [-24,1e-3,1e-3,1e-3], args=(spect_wave, spect_spec, spect_err_, temp_func), epsfcn=1e-6, full_output = True, ftol=1e-6, gtol=1e-6)
                
                print(a[0])
                
                if False:
                    plt.figure()
                    plt.plot(spect_wave,a[2]['fvec'])
                    plt.show()
                #import pdb; pdb.set_trace()

                #!!! To be neatened. All velocities should be saved in a 2D array !!!
                if a[0][0] != -24 and not (a[1] is None):
                   rvs.append(a[0][0])
                   #Multiply the error but the square root of chi-squared. !!! Why is chi-squared so high? !!!
                   rv_errs.append(np.sqrt(np.mean(a[2]['fvec']**2))*np.sqrt(a[1][0,0]))

            rvs = np.array(rvs)
            rv_errs = np.array(rv_errs)
            print(np.median(rvs))
            weights = 1/rv_errs**2
            wtmn_rv = np.sum(weights*rvs)/np.sum(weights)
            wtmn_rv_err = 1/np.sqrt(np.sum(weights))
            print("Weighted mean RV (km/s): {:.4f} +/- {:.4f}".format(wtmn_rv, wtmn_rv_err))
            #The mean square error will be significantly larger than 1 (e.g. more than 1.5) if fiber to fiber 
            #variations are determined by more than just random errors.
            print("MSE: {:.2f}".format(np.mean((wtmn_rv - rvs)**2/rv_errs**2)))
            if False:
                plt.figure()
                plt.errorbar(np.arange(len(rvs))+1, rvs, rv_errs, fmt='.')
                plt.title('Order:' + str(i))
                plt.xlabel('Fiber num')
                plt.ylabel('Velocity (km/s)')
                plt.show()
            
            
            BC_t, BC_star = barycentric_correction('11dec30096o.fits',obs_file_path[48:58]+'o.fits','191211',fitting_dates[fit_index])
            
            print('True Velocity ', BC_star*c.c.to(u.km/u.s).value)
            print('Velocity Difference (m/s) ', wtmn_rv*1000 - BC_star*c.c.to(u.m/u.s).value)

            print('Percentage Error ', 100*(wtmn_rv - BC_star*c.c.to(u.km/u.s).value)/(BC_star*c.c.to(u.km/u.s).value))
            velocity_err[file_ind, order_ind] = (wtmn_rv*1000 - BC_star*c.c.to(u.m/u.s).value)
            order_ind += 1
        file_ind += 1
    for order in range(36):        
        plt.figure()   
        plt.plot([1,2],velocity_err[0:2,order],'.',marker=".", markersize=10,label = '11dec2019')
        plt.plot([3,4,5],velocity_err[2:5,order],'.',marker=".", markersize=10,label = '12dec2019')
        plt.plot([6,7],velocity_err[5:7,order],'.',marker=".", markersize=10,label = '13dec2019')
        plt.plot([8,9,10],velocity_err[7:10,order],'.',marker=".", markersize=10,label = '14dec2019')
        plt.plot( [11,12,13],velocity_err[10:13,order],'.',marker=".", markersize=10,label = '15dec2019')
        plt.title(orders[order])
        plt.ylabel('Velocity Error (m/s)')
        plt.xlabel('Observation Number')
        plt.legend(loc='best')
        plt.show()


