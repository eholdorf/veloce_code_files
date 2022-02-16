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
log_scale_interpolation('/priv/avatar/velocedata/Data/spec_211202/191211/11dec30096oi_extf.fits','/priv/avatar/velocedata/Data/spec_211202/191211/11dec30096oi_extf.fits')
