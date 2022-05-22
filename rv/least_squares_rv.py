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

def generate_rvs(star_name, date, template_path, int_guess = 0.1, telluric_depth_limit = 0.05):   
    
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

    orders = list(range(39)) 
    #orders = [6,7,13,14,17,25,26,27,28,30,31,33,34,35,36,37]
    rvs = np.zeros((len(files),len(orders),19))
    rv_errs = np.zeros((len(files),len(orders),19))
    mses = np.empty((len(files),len(orders),19))
    med_flux = np.zeros((len(files),len(orders),19))

    wtmn_rv = np.empty((len(files),len(orders)))
    wtmn_rv_err = np.empty((len(files),len(orders)))
    velocity_errors = np.empty((len(files),len(orders)))
    total_rv = np.empty(len(files))
    total_rv_error = np.empty(len(files))
    
    BCs = np.empty(len(files))
    MJDs = np.empty(len(files))

    for fit_index,fits in enumerate(files):
        obs_file_path = '/priv/avatar/ehold13/obs_corrected/'+star_name+'/'+fits[0:10]+'_corrected.fits'
        observation = pyfits.open(obs_file_path)
        
        orig_file = pyfits.getheader('/priv/avatar/velocedata/Data/spec_211202/' + date + '/' + fits[0:10] + 'oi_extf.fits')
        MJDs[fit_index] = orig_file['UTMJD']
        
        BC_t, BC_star = barycentric_correction('11dec30096o.fits',fits[0:10]+'o.fits','191211',date)
        BCs[fit_index] = BC_star*c.c.to(u.km*u.s**-1).value
                
        print('Fitting Observation '+fits+', '+str(int(fit_index)+1)+'/'+str(len(files)))
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
                
                spect = spectrum[830:3200,order,fibre]
                mask = np.isnan(spect)
                spect = spect[~mask]
                wave = wavelength[830:3200,order,fibre][~mask].astype(np.float64)
                log_wave = np.log(wave)
                err = error[830:3200,order,fibre][~mask]
                


                med_flux[fit_index,order_index,fibre_index] = np.median(spect)
                
                mask = np.isnan(spect)
                scale = np.median(spect[~mask])
                #print(scale)
                spect /= scale
                err /= scale
                scaled_median = med_flux[fit_index,order_index,fibre_index]/scale
                
             
                if 0==len(log_wave):
                    continue
                    
                # set error to infinity in in deep telluric
                for wave_index, lam in enumerate(wave):
                    for index, line in enumerate(lines):
                        if line - widths[index] < lam and lam < line + widths[index] and depths[index] > telluric_depth_limit:
                            err[wave_index] = np.inf
                        
                        if abs(spect[wave_index])>3*abs(scaled_median):
                            err[wave_index] = np.inf

                
                if np.isinf(err).all():
                    continue            
                initial_cond = [int_guess,1e-3,1e-3,1e-3]
                a = optimise.least_squares(rv_fitting_eqn,x0 = initial_cond, args=(log_wave, spect, err, temp_spec, temp_lwave[0], temp_dlwave), \
                    jac=rv_jac, method='lm')
                if order == 130:
                    plt.figure()
                    plt.plot(wave,spect,label='original')
                    plt.plot(wave,rv_fitting_eqn(a.x,log_wave, spect, err, temp_spec, temp_lwave[0], temp_dlwave, return_spec = True),label = 'fitted')
                    plt.legend()
                    plt.show()    
                    
                cov = np.linalg.inv(np.dot(a.jac.T,a.jac))    
                if a.success: 
                    rvs[fit_index,order_index,fibre_index] = a.x[0]
                    mse = np.mean(a.fun**2) 
                    mses[fit_index,order_index,fibre_index] = mse
                    rv_errs[fit_index,order_index,fibre_index] = np.sqrt(mse)*np.sqrt(cov[0,0])
                    
                
            weights = 1/rv_errs[fit_index,order_index,:]**2
            wtmn_rv[fit_index,order_index] = np.nansum(weights*rvs[fit_index,order_index,:])/np.nansum(weights)
            wtmn_rv_err[fit_index,order_index] = 1/np.sqrt(np.nansum(weights))
            
                
            print(order_index,wtmn_rv[fit_index,order_index]*1000,'+/-',wtmn_rv_err[fit_index,order_index]*1000, 'm/s')
            
            velocity_errors[fit_index, order_index] = (wtmn_rv[fit_index,order_index]*1000 - BC_star*c.c.to(u.m/u.s).value)
    
        total_weights = 1/wtmn_rv_err[fit_index,:]**2
        for i,elem in enumerate(total_weights):
            if np.isinf(elem):
                total_weights[i] = 0
        total_rv[fit_index] = np.nansum(total_weights*wtmn_rv[fit_index,:])/np.nansum(total_weights)
        total_rv_error[fit_index] = 1/np.sqrt(np.nansum(total_weights))
        print('total', total_rv[fit_index]*1000, '+/-',total_rv_error[fit_index]*1000, 'm/s')
            
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
                
                
            
