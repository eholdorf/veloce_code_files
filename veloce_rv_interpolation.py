import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy.signal import correlate
from astropy.table import Table
import get_observations
import astropy.constants as c
import astropy.units as u
from barycorrpy import get_BC_vel

def log_scale_interpolation(template_obs, star_obs,k=5):
    """
    Description
    -----------
    This code will put a template spectrum and star spectrum onto a log wavelength scale.
    
    Parameters
    ----------
    template_spectrum_dir : type - string
        directory for the template data
        
    star_spectrum_dir : type - string
        directory for the data for observed star     
    
    k : type - int
        order of the spline between interpolation points   
    
    Returns
    -------
    all_log_w : type - list of lists
        list of log scale wavelength grid for each fibre
    
    t_logflux : type - list of lists
        list of flux for template on log wavelength grid for each fibre
    
    s_logflux : type - list of lists
        list of flux for star on log wavelength grid for each fibre
    """
    veloce_obs = Table.read('veloce_observations.fits')
    template_i = 0
    template_j = 0
    star_i = 0
    star_j = 0
    for star in veloce_obs:
        template_j = 0
        star_j = 0
        for obs in star[7]:
            if obs.decode("utf-8") == template_obs:
                template_fits_row = veloce_obs[template_i]
                template_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+veloce_obs[template_i][8][template_j].decode("utf-8")+'/'+obs.decode("utf-8")[0:10]+'oi_extf.fits'
            if obs.decode("utf-8") == star_obs:
                star_fits_row = veloce_obs[star_i]
                star_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+veloce_obs[star_i][8][star_j].decode("utf-8")+'/'+obs.decode("utf-8")[0:10]+'oi_extf.fits'
                
            template_j += 1
            star_j += 1
        template_i += 1 
        star_i += 1   
    # read in the data 
    template = pyfits.open(template_spectrum_dir)
    star = pyfits.open(star_spectrum_dir)
    # apply barycentric velocity correction
    template_BC = get_BC_vel(template_fits_row[6]+2400000.5, ra = template_fits_row[1][0], dec = template_fits_row[1][1], obsname = 'SSO')
    template_delta_lambda = template_BC[0][0]*u.m*u.s**-1/c.c
        
    star_BC = get_BC_vel(star_fits_row[6]+2400000.5, ra = star_fits_row[1][0], dec = star_fits_row[1][1], obsname = 'SSO')
    star_delta_lambda = star_BC[0][0]*u.m*u.s**-1/c.c
    # store data for all fibres
    all_log_w = np.zeros((22600,40))
    
    all_t_logflux = np.zeros((22600,40,19))
    all_t_logerrflux = np.zeros((22600,40,19))
        
    all_s_logflux = np.zeros((22600,40,19))
    all_s_logerrflux = np.zeros((22600,40,19))
    
    # set the wavelength scale for a single fibre
    t_logflux = np.zeros((22600,40))
    t_logerrflux = np.zeros((22600,40))
    
    for order in range(len(template[1].data[0,:,4])):
        # make a mask to remove all of the NaN values in spectrum for template and star
        t_mask = np.isnan(template[0].data[:,order,4])
        # extract the wavelengths and flux values which aren't NaN for template and star
        t_w = template[1].data[:,order,4][~t_mask]
        t_w += template_delta_lambda.value*t_w
        t_f = template[0].data[:,order,4][~t_mask]
        t_e = template[2].data[:,order,4][~t_mask]
        # create a log scale from min wavelength to max wavelength of template with 22600 points and add this to the overall wavelength scale
        new_log_w = np.min(t_w)*np.exp(np.log(np.max(t_w)/np.min(t_w))/22600*np.arange(22600))
        all_log_w[:,order] = new_log_w
        #generate a function that will interpolate the flux for new wavelength scale
        t_poly = InterpolatedUnivariateSpline(t_w,t_f,k=k)
        t_err_poly = InterpolatedUnivariateSpline(t_w,t_e,k=k)
        # evaluate the flux with the new wavelength scale
        t_logflux[:,order] = t_poly(new_log_w)
        t_logerrflux[:,order] = t_err_poly(new_log_w)
    all_t_logerrflux[:,:,0] = t_logerrflux
    all_t_logflux[:,:,0] = t_logflux
    
    
    # only choose stellar fibres
    for fibre in range(5,23):
        # define wavelength scale for each fibre
        t_logflux = np.zeros((22600,40))
        t_logerrflux = np.zeros((22600,40))
        for order in range(len(template[1].data[0,:,fibre])):
            # make a mask to remove all of the NaN values in spectrum for template and star
            t_mask = np.isnan(template[0].data[:,order,fibre])
            # extract the wavelengths and flux values which aren't NaN for template and star
            t_w = template[1].data[:,order,fibre][~t_mask]
            t_w += template_delta_lambda.value*t_w
            t_f = template[0].data[:,order,fibre][~t_mask]
            t_e = template[2].data[:,order,fibre][~t_mask]

            #generate a function that will interpolate the flux for new wavelength scale
            t_poly = InterpolatedUnivariateSpline(t_w,t_f,k=k)
            t_err_poly = InterpolatedUnivariateSpline(t_w,t_e,k=k)
            # evaluate the flux with the new wavelength scale
            
            t_logflux[:,order] = t_poly(all_log_w[:,order])
            t_logerrflux[:,order] = t_err_poly(all_log_w[:,order])
            
        all_t_logflux[:,:,fibre-4] = t_logflux
        all_t_logerrflux[:,:,fibre-4] = t_logerrflux
        
    
    # only choose stellar fibres
    for fibre in range(4,23):
        # define wavelength scale for each fibre
        s_logflux = np.zeros((22600,40))
        s_logerrflux = np.zeros((22600,40))
        for order in range(len(template[1].data[0,:,fibre])):
            # make a mask to remove all of the NaN values in spectrum for template and star
            s_mask = np.isnan(star[0].data[:,order,fibre])
            # extract the wavelengths and flux values which aren't NaN for template and star
            s_w = star[1].data[:,order,fibre][~s_mask]
            s_w += star_delta_lambda.value*s_w
            s_f = star[0].data[:,order,fibre][~s_mask]
            s_e = star[2].data[:,order,fibre][~s_mask]
            
            #generate a function that will interpolate the flux for new wavelength scale
            s_poly = InterpolatedUnivariateSpline(s_w,s_f,k=k)
            s_err_poly = InterpolatedUnivariateSpline(s_w,s_e,k=k)
            # evaluate the flux with the new wavelength scale
            s_logflux[:,order] = s_poly(all_log_w[:,order])
            s_logerrflux[:,order] = s_err_poly(all_log_w[:,order])
            
        all_s_logflux[:,:,fibre-4] = s_logflux
        all_s_logerrflux[:,:,fibre-4] = s_logerrflux
        
    return all_log_w, all_t_logflux, all_t_logerrflux, all_s_logflux, all_s_logerrflux

#testing code for a star  
#w,t,terr,s,serr = log_scale_interpolation('11dec30096o.fits','11dec30096o.fits')

# test make_template
file_path = '/priv/avatar/velocedata/Data/spec_211202/'

# Tau Ceti (HD10700) template from dec 2019

TC_observation_dir = ['191211/11dec30096oi_extf.fits', '191211/11dec30097oi_extf.fits', '191212/12dec30132oi_extf.fits', '191212/12dec30133oi_extf.fits','191212/12dec30134oi_extf.fits', '191213/13dec30076oi_extf.fits', '191213/13dec30077oi_extf.fits', '191214/14dec30066oi_extf.fits', '191214/14dec30067oi_extf.fits', '191214/14dec30068oi_extf.fits', '191215/15dec30097oi_extf.fits', '191215/15dec30098oi_extf.fits', '191215/15dec30099oi_extf.fits']


def generate_template(file_paths):
    """
    Description
    -----------
    This code will create a template spectrum on a evenly spaced log wavelength scale.
    
    Parameters
    ----------
    file_paths : type - list of strings
        list of paths to each observation
    
    Returns
    -------
    wavelength : type - list of lists
        the log wavelength scale
    
    template : type - list of lists
        the weighted average (between fibres) template spectrum
    
    error_template : type - list of lists
        the error for each point on the template
    """    
    # generate a blank spectrum template to fill
    wavelength, flux_t, flux_t_err, flux_s, flux_s_err = log_scale_interpolation(file_paths[0])
    dd = pyfits.open(file_paths[0])
    template_spectrum = np.zeros([np.shape(dd[0])[0],np.shape(dd[0])[1],np.shape(dd[0])[2]], dtype = object)
    template_spectrum_error = np.zeros([np.shape(dd[0])[0],np.shape(dd[0])[1],np.shape(dd[0])[2]], dtype = object)
    
    # iterate through each observation that is wanted in the template
    num_good_pixels = np.ones([np.shape(dd[0])[0],np.shape(dd[0])[1],np.shape(dd[0])[2]],dtype = object)
    for obs in file_paths:
        print(':0')
        dd = pyfits.open(obs)
        # iterate through the fibres
        good_pixels = np.zeros([np.shape(dd[0])[0],np.shape(dd[0])[1],np.shape(dd[0])[2]],dtype = object)
        error_good_pixels = np.zeros([np.shape(dd[0])[0],np.shape(dd[0])[1],np.shape(dd[0])[2]],dtype = object)
        for fibre in range(19):
            # iterate through the orders
            for order in range(40):
                # pull out spectrum
                spectrum = dd[0].data[:,order,fibre+4]
                error = dd[2].data[:,order,fibre+4]
                i = 0
                while i < len(spectrum):
                    good_pixels[i, order, fibre+4] = spectrum[i]
                    error_good_pixels[i,order,fibre+4] = error[i]
                    num_good_pixels[i,order,fibre+4] += 1
                            
                    i += 1
        template_spectrum += good_pixels
        template_spectrum_error += error_good_pixels
    # find the median fibre height to check for bad pixels, only include the stellar fibres
    template_spectrum /=num_good_pixels
    template_spectrum_error /= num_good_pixels
    median_fibre_spectrum = np.median(template_spectrum[:,:,4:23]/num_good_pixels[:,:,4:23],2)
    median_error_spectrum = np.median(template_spectrum_error[:,:,4:23]/num_good_pixels[:,:,4:23],2)
    
    # for each fibre, find the median difference between it and median_fibre_spectrum and remove pixels with a difference much larger than this
    template = np.zeros([np.shape(dd[0])[0],np.shape(dd[0])[1]],dtype = object)
    error = np.zeros([np.shape(dd[0])[0],np.shape(dd[0])[1]],dtype = object)
    weights = np.ones([np.shape(dd[0])[0],np.shape(dd[0])[1]],dtype = object)
    
    for fibre in range(19):
        diff = [abs(template_spectrum[:,:,4+fibre] - median_fibre_spectrum[:,:])]
        med_diff = np.median(diff)
        # the distance from the median difference for each point
        diff = abs(diff - med_diff)[0,:,:]       
        
        
        # for each point in each fibre keep in template if difference to median spectrum - median distance/median distance is less than 1.1
        for order in range(40):
            for wave in range(np.shape(diff)[0]):
                # only include a value if it is within __ of the median value
                if diff[wave,order]/med_diff<=2:
                    # add one to the weight for this wavelength to compute the weighted average
                    weights[wave,order] += 1
                    # add spectrum value to this wavelength position in template
                    template[wave,order] += template_spectrum[wave,order,fibre+4]
                    error[wave,order] += template_spectrum_error[wave,order,fibre+4]
                    
    # define the wavelength scale, choose any fibre as all the same
    wavelength = dd[1].data[:,:,5].copy()
    template /= weights
    error /= weights
    # set all 0 values to NaN and values which have a low signal to noise
    for order in range(40):
        for wave in range(np.shape(diff)[0]):
            if (template[wave,order]==0) | (template[wave,order]<3*error[wave,order]):
                template[wave,order] = np.nan
                wavelength[wave,order] = np.nan
    plt.figure()
    plt.plot(wavelength,template)
    plt.figure()
    plt.plot(wavelength,median_fibre_spectrum)
    plt.show()
    
    plt.plot(wavelength,diff/med_diff-template_spectrum_error[:,:,fibre])
    plt.show()
    plt.plot(wavelength,weights)
    plt.show()   
    # return the template spectrum with weighted average
    return wavelength, template, error
        
testing_temp_files = ['11dec30096o.fits', '11dec30097o.fits', '12dec30132o.fits', '12dec30133o.fits', '12dec30134o.fits', '13dec30076o.fits', '13dec30077o.fits', '14dec30066o.fits', '14dec30067o.fits', '14dec30068o.fits', '15dec30097o.fits', '15dec30098o.fits', '15dec30099o.fits']

#ob = [file_path+TC_observation_dir[i] for i in range(len(TC_observation_dir))]

#w, s = generate_template(ob)

def calc_rv_corr(file_path, observation_dir, star_spectrum_dir, k=5):
    # generate a template and wavelength scale
    template_wave, template_spec = make_template(file_path,observation_dir)
    # interpolate the star spectrum onto the same wavelength scale as above
    wavelength, template, star_spec = log_scale_interpolation(file_path+observation_dir[0],star_spectrum_dir)
    # calculate the cross-correlation between each order of each fibre between the star and the template
    correlation = []
    for fibre in range(np.shape(template_spec)[0]):
        order_cor = []
        for order in range(np.shape(template_spec)[1]):
            # perfrom cross-correlation between template and spectrum
            cor = correlate(star_spec[fibre][order], template_spec[fibre][order], mode = 'same',method = 'auto')
            order_cor.append(cor)
        correlation.append(order_cor)
    return correlation
    
   
#calc_rv_corr(file_path, TC_observation_dir, '/priv/avatar/velocedata/Data/spec_211202/191211/11dec30096oi_extf.fits')
    
