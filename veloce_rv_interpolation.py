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
    template_obs_num = 0
    star_i = 0
    star_j = 0
    star_obs_num = 0
    for star in veloce_obs:
        template_j = 0
        star_j = 0
        for obs in star[7]:
            if obs.decode("utf-8") == template_obs:
                template_fits_row = veloce_obs[template_i]
                template_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+veloce_obs[template_i][8][template_j].decode("utf-8")+'/'+obs.decode("utf-8")[0:10]+'oi_extf.fits'
                template_obs_num = template_j
            if obs.decode("utf-8") == star_obs:
                star_fits_row = veloce_obs[star_i]
                star_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+veloce_obs[star_i][8][star_j].decode("utf-8")+'/'+obs.decode("utf-8")[0:10]+'oi_extf.fits'
                star_obs_num = star_j
                
            template_j += 1
            star_j += 1
        template_i += 1 
        star_i += 1   
    # read in the data 
    template = pyfits.open(template_spectrum_dir)
    star = pyfits.open(star_spectrum_dir)
  
    # apply barycentric velocity correction
    template_BC = get_BC_vel(template_fits_row[6][template_obs_num]+2400000.5, ra = template_fits_row[1][0], dec = template_fits_row[1][1], obsname = 'SSO')
    template_delta_lambda = template_BC[0][0]*u.m*u.s**-1/c.c
        
    star_BC = get_BC_vel(star_fits_row[6][star_obs_num]+2400000.5, ra = star_fits_row[1][0], dec = star_fits_row[1][1], obsname = 'SSO')
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
    file_paths : type - list of bytes
        list of fits files with observations to make into template from veloce_observations.fits
    
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
    wavelength, flux_t, flux_t_err, flux_s, flux_s_err = log_scale_interpolation(file_paths[0],file_paths[0])
    #dd = pyfits.open(file_paths[0])
    template_spectrum = np.zeros([np.shape(flux_t)[0],np.shape(flux_t)[1],np.shape(flux_t)[2]], dtype = object)
    template_spectrum_error = np.zeros([np.shape(flux_t)[0],np.shape(flux_t)[1],np.shape(flux_t)[2]], dtype = object)
    template_spectrum_one_on_error = np.zeros([np.shape(flux_t)[0],np.shape(flux_t)[1],np.shape(flux_t)[2]], dtype = object)
   
    # iterate through each observation that is wanted in the template
    num_good_pixels = np.ones([np.shape(flux_t)[0],np.shape(flux_t)[1],np.shape(flux_t)[2]],dtype = object)
    for obs in file_paths:
        print(':0')
        wavelength, flux_t, flux_t_err, flux_s, flux_s_err = log_scale_interpolation(file_paths[0],obs)
        # iterate through the fibres
        good_pixels = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1],np.shape(flux_s)[2]],dtype = object)
        error_good_pixels = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1],np.shape(flux_s)[2]],dtype = object)
        one_over_error_good_pixels = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1],np.shape(flux_s)[2]],dtype = object)
        for fibre in range(19):
            # iterate through the orders
            for order in range(40):
                # pull out spectrum
                spectrum = flux_s[:,order,fibre]
                scale = np.median(spectrum)
                spectrum /= scale
                error = flux_s_err[:,order,fibre]/scale
                
                i = 0
                while i < len(spectrum):
                    good_pixels[i, order, fibre] = spectrum[i]/error[i]**2
                    error_good_pixels[i,order,fibre] = error[i]
                    one_over_error_good_pixels[i,order,fibre] = 1/error[i]**2
                    num_good_pixels[i,order,fibre] += 1
                            
                    i += 1
        template_spectrum += good_pixels
        template_spectrum_error += error_good_pixels
        template_spectrum_one_on_error += one_over_error_good_pixels
    # find the median fibre height to check for bad pixels, only include the stellar fibres
    template_spectrum_error /= num_good_pixels
    template_spectrum_one_on_error /= num_good_pixels
    template_spectrum /=num_good_pixels*template_spectrum_one_on_error
    template_spectrum_error = (1/template_spectrum_one_on_error)**0.5
    
    plt.figure()
    plt.plot(wavelength,template_spectrum[:,:,10])
    
    median_fibre_spectrum = np.median(template_spectrum,2)
    median_error_spectrum = np.median(template_spectrum_error,2)
    
    # for each fibre, find the median difference between it and median_fibre_spectrum and remove pixels with a difference much larger than this
    template = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1]])
    error = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1]])
    one_on_error = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1]])
    weights = np.ones([np.shape(flux_s)[0],np.shape(flux_s)[1]])
    
    for fibre in range(19):
        print(':|')
        diff = [abs(template_spectrum[:,:,fibre] - median_fibre_spectrum[:,:])]
        med_diff = np.median(diff)
        # the distance from the median difference for each point
        diff = abs(diff - med_diff)[0,:,:]
        
        # for each point in each fibre keep in template if difference to median spectrum - median distance/median distance is less than 1.1
        for order in range(np.shape(diff)[1]):
            scale = np.median(template_spectrum[:,order,fibre])
            
            for wave in range(np.shape(diff)[0]):
                # only include a value if it is within __ of the median value
                if (diff[wave,order]/med_diff<=2) & (template_spectrum[wave,order,fibre]>3*template_spectrum_error[wave,order,fibre]):
                    # add one to the weight for this wavelength to compute the weighted average
                    weights[wave,order] += 1
                    # add spectrum value to this wavelength position in template
                    template[wave,order] += (template_spectrum[wave,order,fibre]/scale)/(template_spectrum_error[wave,order,fibre]**2/scale**2)
                    error[wave,order] += template_spectrum_error[wave,order,fibre]/scale
                    one_on_error[wave,order] += scale**2/template_spectrum_error[wave,order,fibre]**2
                   
                    
    # define the wavelength scale, choose any fibre as all the same
    error /= weights
    one_on_error /= weights
    template /= weights*one_on_error
    error = (1/one_on_error)**0.5
    # set all 0 values to NaN and values which have a low signal to noise
    for order in range(40):
        for wave in range(np.shape(diff)[0]):
            if (template[wave,order]==0)| (template[wave,order]<3*error[wave,order]):
                template[wave,order] = np.nan
                wavelength[wave,order] = np.nan
    plt.figure()
    plt.plot(wavelength,template)
    plt.show()
  
    # return the template spectrum with weighted average
    return wavelength, template, error
        
testing_temp_files = ['11dec30096o.fits', '11dec30097o.fits', '12dec30132o.fits', '12dec30133o.fits', '12dec30134o.fits', '13dec30076o.fits', '13dec30077o.fits', '14dec30066o.fits', '14dec30067o.fits', '14dec30068o.fits', '15dec30097o.fits', '15dec30098o.fits', '15dec30099o.fits']

w, s, e = generate_template(testing_temp_files)

#primary_hdu = pyfits.PrimaryHDU(s)
#image_hdu = pyfits.ImageHDU(w)
#image_hdu2 = pyfits.ImageHDU(e)
#hdul = pyfits.HDUList([primary_hdu, image_hdu, image_hdu2])
#hdul.writeto('Tau_Ceti_Template_dec2019.fits')

def telluric_correction(obs_night):
    """
    Description
    -----------
    This code will provide a telluric correction for the nights in the template
    
    Parameters
    ----------
    obs_night : type - byte
        night in format yymmdd for observation to find telluric standard
    
    Returns
    -------
    bstars : type - list of lists
        list containing all observations which were BSTARS from given obs_night, each element is a list containing the name of the object, the fits file and the directory
    """ 
    dd = Table.read('veloce_observations.fits')
    dd = dd[dd['obs_type']=='BSTAR']
    bstars = []
    for obs in dd:
         i = 0
         for dire in obs[8]:
            i +=1
            if dire == obs_night:
                bstars.append([obs[0],obs[7][i],obs[8][i]])
    telluric_star = bstars[-1]
    wavelength, flux_t, flux_t_err, flux_s, flux_s_err = log_scale_interpolation(telluric_star[1].decode('utf-8'),telluric_star[1].decode('utf-8'))
   
    telluric_spec = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1]])
    error_spec = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1]])
    over_error_spec = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1]])
    weights = np.ones([np.shape(flux_s)[0],np.shape(flux_s)[1]])

    for fibre in range(19):
        print(fibre)
        for order in range(40):
            scale = np.median(flux_s[:,order,fibre])
            spectrum = flux_s[:,order,fibre]/scale
            error = flux_s_err[:,order,fibre]/scale
            for i in range(22600):
                if flux_s[i,order,fibre]> 3*flux_s_err[i,order,fibre]:
                    telluric_spec[i,order] += (flux_s[i,order,fibre]/scale)/(flux_s_err[i,order,fibre]/scale)**2
                    error_spec[i,order] += error[i]
                    over_error_spec[i,order] += scale**2/flux_s_err[i,order,fibre]**2
                    weights[i,order] += 1
    telluric_spec /= over_error_spec*weights
    over_error_spec /= weights
    error_spec = (1/over_error_spec)**0.5
    
    for order in range(40):
        for wave in range(np.shape(telluric_spec)[0]):
            if (telluric_spec[wave,order]==0):# | (telluric_spec[wave,order]< 3*error_spec[wave,order]):
                telluric_spec[wave,order] = np.nan
                wavelength[wave,order] = np.nan
    
    plt.figure()             
    plt.plot(wavelength,telluric_spec)
    plt.figure()
    plt.plot(wavelength,flux_s[:,:,13])
    plt.show()
                
                                     
    return bstars
    
# test telluric_correction
dd = Table.read('veloce_observations.fits')
bstar = telluric_correction(dd[0][8][0])

test = bstar[0]
test_data = pyfits.open('/priv/avatar/velocedata/Data/spec_211202/'+test[2].decode('utf-8')+'/'+test[1].decode('utf-8')[0:10]+'oi_extf.fits')

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
    
