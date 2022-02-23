import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

def log_scale_interpolation(template_spectrum_dir, star_spectrum_dir,k=5):
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

    # read in the data 
    template = pyfits.open(template_spectrum_dir)
    star = pyfits.open(star_spectrum_dir)
    # store data for all fibres
    all_log_w = []
    all_t_logflux = []
    all_s_logflux = []
    # only choose stellar fibres
    for fibre in range(5,23):
        # define wavelength scale for each fibre
        log_w = []
        t_logflux = []
        s_logflux = []
        for order in range(len(template[1].data[0,:,fibre])):
            # make a mask to remove all of the NaN values in spectrum for template and star
            t_mask = np.isnan(template[0].data[:,order,fibre])
            s_mask = np.isnan(star[0].data[:,order,fibre])
            # extract the wavelengths and flux values which aren't NaN for template and star
            t_w = template[1].data[:,order,fibre][~t_mask]
            t_f = template[0].data[:,order,fibre][~t_mask]
            s_w = template[1].data[:,order,fibre][~s_mask]
            s_f = template[0].data[:,order,fibre][~s_mask]
            # create a log scale from min wavelength to max wavelength of template with the same number of points and add this to the overall wavelength scale
            new_log_w = np.min(t_w)*np.exp(np.log(np.max(t_w)/np.min(t_w))/len(t_w)*np.arange(len(t_w)))
            log_w.extend(new_log_w)
            #generate a function that will interpolate the flux for new wavelength scale
            t_poly = InterpolatedUnivariateSpline(t_w,t_f,k=k)
            s_poly = InterpolatedUnivariateSpline(s_w,s_f,k=k)
            # evaluate the flux with the new wavelength scale
            t_logflux.extend(t_poly(new_log_w))
            s_logflux.extend(s_poly(new_log_w))
        all_log_w.append(log_w)
        all_t_logflux.append(t_logflux)
        all_s_logflux.append(s_logflux)
    return all_log_w, all_t_logflux, all_s_logflux

# testing code for a star  
#log_scale_interpolation('/priv/avatar/velocedata/Data/spec_211202/191211/11dec30096oi_extf.fits','/priv/avatar/velocedata/Data/spec_211202/191211/11dec30096oi_extf.fits')

def make_template(file_path,observation_dir):
    """
    Description
    -----------
    This code will create a template spectrum on a evenly spaced log wavelength scale.
    
    Parameters
    ----------
    file_path : type - string
        path to where each of the elements of observation_dir are
    
    observation_dir : type - list of strings
       for each observation needed in the template, the file path to each different observation (i.e. rest of file path after file_path input)
    
    Returns
    -------
    log_scale : type - np.array() of lists
        the log wavelength scale for each of the stellar fibres 
    
    template_spectrum : type - np.array() of lists
        the template spectrum for each of the stellar fibres
    """
    # go through all observations but the first, we will use this as the template to move all of the other observations to
    for template in observation_dir[1:]:
        #create the log scale interpolation
        result = log_scale_interpolation(file_path+observation_dir[0],file_path+template)
        log_scale = []
        # if this is the first run, then save the log scale and start the template spectrum and add in the star observation
        if len(log_scale) == 0:
            log_scale = np.array(result[0])
            template_spectrum = np.array(result[1])
            template_spectrum += result[2]
        # if it isn't the first run, then just add the star spectrum to the template
        else:
            template_spectrum += result[2]
    return log_scale, template_spectrum/len(observation_dir)

# test make_template
file_path = '/priv/avatar/velocedata/Data/spec_211202/'

# Tau Ceti (HD10700) template from dec 2019

TC_observation_dir = ['191211/11dec30096oi_extf.fits', '191211/11dec30097oi_extf.fits', '191212/12dec30132oi_extf.fits', '191212/12dec30133oi_extf.fits','191212/12dec30134oi_extf.fits', '191213/13dec30076oi_extf.fits', '191213/13dec30077oi_extf.fits', '191214/14dec30066oi_extf.fits', '191214/14dec30067oi_extf.fits', '191214/14dec30068oi_extf.fits', '191215/15dec30097oi_extf.fits', '191215/15dec30098oi_extf.fits', '191215/15dec30099oi_extf.fits']

wave, spect = make_template(file_path, TC_observation_dir)
plt.plot(wave[0], spect[0],'.')
plt.show()
    
