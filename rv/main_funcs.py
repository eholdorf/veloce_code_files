import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy.signal import correlate
from astropy.table import Table
from . import get_observations
import astropy.constants as c
import astropy.units as u
try:
    from barycorrpy import get_BC_vel
except:
    raise UserWarning("No Barycorrpy! You need to install this using 'pip install barycorrpy'")
import scipy.optimize as opt
from . import utils

def log_scale_interpolation(template_obs, star_obs, template_date, star_date, k=5, BC = False, num_points = 22600):
    """
    Description
    -----------
    This code will put a template spectrum and star spectrum onto a common evenly log spaced wavelength grid.
    
    Parameters
    ----------
    template_obs: type - string
        fits file name from veloce_observations.fits that the wavelength scale is being shifted to (e.g. '11dec30096o.fits')
        
    star_obs : type - string
        fits file name from veloce_observations.fits that will have the wavelength scale shifted to template_obs (e.g. '11dec30096o.fits')  
    
    template_date: type - string
        date that the template observation was taken in the form yymmdd
        
        
    star_date : type - string
        date that the star observation was taken in the form yymmdd
          
    
    k : type - int (optional : default = 5)
        order of the spline between interpolation points, choose from 1, 2, 3, 4 or 5
        
    BC : type - boolean (optional : default = False)
        True if want to also apply a barycentric velocity correction
        False if don't want to apply a barycentric velocity correction
    
    num_points : type - int (optional : default = 22 600)
        This is the number of points desired in the final interpolation for each order.
    
    Returns
    -------
    all_log_w : type - numpy nd-array 
        evenly log scaled wavlength grid for the stellar fibres which has shape = wavelength x order x fibre = num_points x 40 x 19
    
    t_logflux : type - numpy nd-array
        interpolated flux for template_obs which has shape = wavelength x order x fibre = num_points x 40 x 19
        
    t_logerrflux : type - numpy nd-array
        interpolated flux error for template_obs which has shape = wavelength x order x fibre = num_points x 40 x 19
    
    s_logflux : type - numpy nd-array
        interpolated flux for star_obs which has shape = wavelength x order x fibre = num_points x 40 x 19
    
    s_logerrflux : type - numpy nd-array
        interpolated flux error for star_obs which has shape = wavelength x order x fibre = num_points x 40 x 19
        
    airmasses : type - list
        element 0 is the airmass for template_obs observation
        element 1 is the airmass for star_obs obervation
    """
    # find the observation in the veloce_observations.fits file and store the location in the MSO storage and the airmass
    veloce_obs = Table.read('/home/ehold13/veloce_scripts/veloce_observations.fits')
    if BC:
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
                if obs.decode("utf-8") == template_obs and veloce_obs[template_i][8][template_j].decode('utf-8') == template_date:
                    template_fits_row = veloce_obs[template_i]
                    template_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+veloce_obs[template_i][8][template_j].decode("utf-8")+'/'+obs.decode("utf-8")[0:10]+'oi_extf.fits'
                    template_obs_num = template_j
                    airmass_template = pyfits.open(template_spectrum_dir)[0].header['AIRMASS']
                    
                if obs.decode("utf-8") == star_obs and veloce_obs[star_i][8][star_j].decode('utf-8') == star_date:
                    star_fits_row = veloce_obs[star_i]
                    star_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+veloce_obs[star_i][8][star_j].decode("utf-8")+'/'+obs.decode("utf-8")[0:10]+'oi_extf.fits'
                    star_obs_num = star_j
                    airmass_star = pyfits.open(star_spectrum_dir)[0].header['AIRMASS']
                    
                template_j += 1
                star_j += 1
            template_i += 1 
            star_i += 1   
    else:
        template_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+template_date+'/'+template_obs[0:10]+'oi_extf.fits'
        star_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+star_date+'/'+star_obs[0:10]+'oi_extf.fits'
    # read in the data with found directory
    template = pyfits.open(template_spectrum_dir)
    airmass_template = template[0].header['AIRMASS']
    star = pyfits.open(star_spectrum_dir)
    airmass_star = star[0].header['AIRMASS']
    
    
    # apply barycentric velocity correction
    if BC:
        # add 2400000.5 to convert from MJD to JD
        template_BC = get_BC_vel(template_fits_row[6][template_obs_num]+2400000.5, ra = template_fits_row[1][0], dec = template_fits_row[1][1], obsname = 'SSO')
        template_delta_lambda = template_BC[0][0]*u.m*u.s**-1/c.c
            
        star_BC = get_BC_vel(star_fits_row[6][star_obs_num]+2400000.5, ra = star_fits_row[1][0], dec = star_fits_row[1][1], obsname = 'SSO')
        star_delta_lambda = star_BC[0][0]*u.m*u.s**-1/c.c
#    # if no barycentric correction, then make the shift equal to 0
#    else:
#        template_delta_lambda = 0*u.m/u.s
#        star_delta_lambda = 0*u.m/u.s
    # make numpy nd-arrays to store data for all fibre data
    all_log_w = np.zeros((num_points,40))
    
    all_t_logflux = np.zeros((num_points,40,19))
    all_t_logerrflux = np.zeros((num_points,40,19))
        
    all_s_logflux = np.zeros((num_points,40,19))
    all_s_logerrflux = np.zeros((num_points,40,19))
    
    # set the wavelength scale for a single fibre
    t_logflux = np.zeros((num_points,40))
    t_logerrflux = np.zeros((num_points,40))
    
    # shift to log wavelength scale for the first fibre for all orders
    for order in range(len(template[1].data[0,:,4])):
        # make a mask to remove all of the NaN values in spectrum for the first stellar fibre in the template
        t_mask = np.isnan(template[0].data[:,order,4])
        # extract the wavelengths and flux values which aren't NaN for first fibre in the template
        t_w = template[1].data[:,order,4][~t_mask]
        t_f = template[0].data[:,order,4][~t_mask]
        t_e = template[2].data[:,order,4][~t_mask]
        
        # apply the barycentric correction
        if BC:
            t_w += template_delta_lambda.value*t_w
        
        # create a log scale from min wavelength to max wavelength of template with num_points points and save this to the overall wavelength scale
        new_log_w = np.min(t_w)*np.exp(np.log(np.max(t_w)/np.min(t_w))/num_points*np.arange(num_points))
        all_log_w[:,order] = new_log_w
        
        #generate a function that will interpolate the flux and flux error for new wavelength scale
        t_poly = InterpolatedUnivariateSpline(t_w,t_f,k=k)
        t_err_poly = InterpolatedUnivariateSpline(t_w,t_e,k=k)
        
        # evaluate the flux and flux error with the new wavelength scale
        t_logflux[:,order] = t_poly(new_log_w)
        t_logerrflux[:,order] = t_err_poly(new_log_w)
    
    # save the flux and flux error data for the first stellar fibre
    all_t_logerrflux[:,:,0] = t_logerrflux
    all_t_logflux[:,:,0] = t_logflux
    
    
    # for the remaining stellar fibres repeat the above process, interpolating onto the wavelength scale of the first fibre
    for fibre in range(5,23):
        # create numpy nd-arrays to save flux and flux error data
        t_logflux = np.zeros((num_points,40))
        t_logerrflux = np.zeros((num_points,40))
        
        for order in range(len(template[1].data[0,:,fibre])):
            # make a mask to remove all of the NaN values in spectrum for this fibre in the template
            t_mask = np.isnan(template[0].data[:,order,fibre])
            # extract the wavelengths and flux values which aren't NaN for this fibre in the template
            t_w = template[1].data[:,order,fibre][~t_mask]
            t_f = template[0].data[:,order,fibre][~t_mask]
            t_e = template[2].data[:,order,fibre][~t_mask]
            
            # apply the barycentric correction
            if BC:
                t_w += template_delta_lambda.value*t_w
            
            #generate a function that will interpolate the flux for new wavelength scale
            t_poly = InterpolatedUnivariateSpline(t_w,t_f,k=k)
            t_err_poly = InterpolatedUnivariateSpline(t_w,t_e,k=k)
            
            # evaluate the flux with the new wavelength scale (from the one developed for the first fibre so all will be the same)
            t_logflux[:,order] = t_poly(all_log_w[:,order])
            t_logerrflux[:,order] = t_err_poly(all_log_w[:,order])
        
        # save the flux and flux error    
        all_t_logflux[:,:,fibre-4] = t_logflux
        all_t_logerrflux[:,:,fibre-4] = t_logerrflux
        
    
    # only choose stellar fibres
    for fibre in range(4,23):
        # define wavelength scale for each fibre
        s_logflux = np.zeros((num_points,40))
        s_logerrflux = np.zeros((num_points,40))
        
        # iterate over each of the orders
        for order in range(len(template[1].data[0,:,fibre])):
            # make a mask to remove all of the NaN values in spectrum for template and star
            s_mask = np.isnan(star[0].data[:,order,fibre])
            # extract the wavelengths and flux values which aren't NaN for template and star
            s_w = star[1].data[:,order,fibre][~s_mask]
            s_f = star[0].data[:,order,fibre][~s_mask]
            s_e = star[2].data[:,order,fibre][~s_mask]
            # apply the barycentric correction
            if BC:
                s_w += star_delta_lambda.value*s_w
                       
            #generate a function that will interpolate the flux for new wavelength scale
            s_poly = InterpolatedUnivariateSpline(s_w,s_f,k=k)
            s_err_poly = InterpolatedUnivariateSpline(s_w,s_e,k=k)
            # evaluate the flux with the new wavelength scale
            s_logflux[:,order] = s_poly(all_log_w[:,order])
            s_logerrflux[:,order] = s_err_poly(all_log_w[:,order])
        
        # save the flux and flux error
        all_s_logflux[:,:,fibre-4] = s_logflux
        all_s_logerrflux[:,:,fibre-4] = s_logerrflux
    
    # save the airmasses for the template and for the star observations
    airmasses = [airmass_template, airmass_star]
        
    return all_log_w, all_t_logflux, all_t_logerrflux, all_s_logflux, all_s_logerrflux, airmasses


def find_telluric_star(obs_night, time, date):
    """
    Description
    -----------
    This code will find a telluric star observation for a given input observation which is : closest to this observation but before it, closest to this observation but after it, or is the closest observation regardless.
    
    Parameters
    ----------
    obs_night : type - string
        fits file name from veloce_observations.fits that a telluric star is required for (e.g. '11dec30096o.fits')
        
    time : type - string
        time wanted for the telluric star relative to the time of obs_night, choose from 'before', 'after' or 'closest'

    date : type - string
        date of obs_night in form yymmdd
    
    Returns
    -------
    science_target : type - list
        this list contains four elements
        element 0 - string : name of the star in the obs_night observation
        element 1 - byte (string) : observation fits file name as in veloce_observations.fits
        element 2 - byte (int) : date of the observation as in veloce_observations.fits
        element 3 - byte (float) : modified julian date for the observation
    
    telluric_star : type - list
        this list contains four elements
        element 0 - string : name of the telluric star found
        element 1 - byte (string) : telluric fits file name as in veloce_observations.fits
        element 2 - byte (int) : date of the telluric star observation as in veloce_observations.fits
        element 3 - byte (float) : modified julian date for the telluric star observation
    """
    # read in the fits file with all of the information for Veloce observations
    dd = Table.read('/home/ehold13/veloce_scripts/veloce_observations.fits')
    # get info for observation want to telluric correct
    for obs in dd:
        for i,fit in enumerate(obs[7]):
            if fit.decode('utf-8') == obs_night and obs[8][i].decode('utf-8')== date:
                # save the (name, fits name, dir, jd)
                science_target = [obs[0],obs[7][i],obs[8][i],obs[6][i]]
                break
                            
    # isolate the BSTAR observations (can use as telluric standards)
    dd = dd[dd['obs_type']=='BSTAR']
    
    # define variable a which depends on what time want the observations (use this as a measure of how far a telluric observation is from the desired time)
    if time == 'before':
        a = -10e10
    else:       
        a = 10e10
    
    # set up a generic list for the telluric star information    
    telluric_star = [0,0,0,a]
    
    # for each BSTAR observation, if observed on the same day and was taken closer in time to the star observation, then save this as the telluric star, otherwise repeat
    for obs in dd:
         i = 0
         # iterate through the observation dates
         for dire in obs[8]:
            #i +=1
            if True:#dire == science_target[2]:
                # if it was observed, then add the star name, fits file and directory to the list of telluric standards for that night
                time_diff = science_target[3] - obs[6][i]
                if time == 'before' and time_diff>0 and time_diff<(science_target[3] - telluric_star[3]):
                    telluric_star = [obs[0],obs[7][i],obs[8][i],obs[6][i]]
                
                elif time == 'after' and time_diff<0 and time_diff>(science_target[3] - telluric_star[3]):
                    telluric_star = [obs[0],obs[7][i],obs[8][i],obs[6][i]]
                
                elif time == 'closest' and abs(time_diff)<abs(science_target[3] - telluric_star[3]):
                    telluric_star = [obs[0],obs[7][i],obs[8][i],obs[6][i]]
            i +=1

    # if couldn't find an observation that fit the time criteria, then repeat this process except look for the closest telluric observation
    if telluric_star == [0,0,0,a]:
        science_target, telluric_star = find_telluric_star(obs_night, 'closest',date)
    return science_target, telluric_star

def telluric_masking(wavelengths, spectrum, order, Bplus = None):
    """
    Description
    -----------
    This code will return the spectrum given, with the absorption lines masked over
    
    Parameters
    ----------
    wavelengths : type - numpy nd-array
        Array of the wavelengths for the spectrum
    
    spectrum : type - numpy nd-array
        Array of the spectrum to mask over
    
    order : type - int
        The order which the spectrum is for
    
    Bplus : type - numpy nd-array (optional - default None)
        The Bplus matrix which is used to do the telluric masking, if None one will be created.
    
    Returns
    -------
    spectrum : type - numpy nd-array
        Corrected telluric spectrum
    
    B_plus : type - numpy nd-array
        The generated or given Bplus matrix
    """
    # read in set of given tellurics and widths
    tellurics = pyfits.open('/home/ehold13/veloce_scripts/Telluric_Lines_And_Widths.fits')
    lines = tellurics[0].data[:,order]
    mask = np.isnan(lines)
    lines = lines[~mask]
    widths = tellurics[1].data[:,order][~mask]
    
    # create an empty list to set as 1 if a wavelength has a telluric line there
    bad_wavelengths = np.zeros(len(wavelengths))
    
    for i,wave in enumerate(wavelengths):
        for j, line in enumerate(lines):
            width = widths[j]
            if line - width < wave and wave < line + width:
                bad_wavelengths[i] = 1
    
    patched_spectrum, B_plus = utils.correct_bad(spectrum,bad_wavelengths, Bplus)
    
    std = np.ceil(len(spectrum)/3900)*10
    gaussian = np.exp(-np.arange(-3*std, 3*std)**2/(2*std**2))
    
    smoothed_patched_spectrum = np.convolve(patched_spectrum,gaussian,mode = 'same')
    
    return spectrum/smoothed_patched_spectrum, B_plus
   

def telluric_correction(obs_night, time, date, scrunch = True, B_plus = None, airmass_corr = True, scale_star = '11dec30096o.fits', scale_date = '191211'):
    """
    Description
    -----------
    This code will provide a telluric correction for the nights in the template
    
    Parameters
    ----------
   
    obs_night : type - string
        fits file name from veloce_observations.fits that a telluric star is required for (e.g. '11dec30096o.fits')
        
    time : type - string
        time wanted for the telluric star relative to the time of obs_night, choose from 'before', 'after' or 'closest'
        
    date : type - string
        date of obs_night in form yymmdd

    scrunch : type - boolean (optional - default True)
        if True will put the observation onto a 22600 log wavelength grid, if False onto 3900 wavelength grid
    
    B_plus : type - numpy nd-array (optional - defualt None)
        matrix to do absorption line patching, if None one will be created

    airmass_corr : type - boolean (optional - default True)
        if True the telluric spectrum will be airmass corrected, if False then it won't be
    
    scale_star : type - string (optional - default '11dec30096.fits')
        fits file from veloce_observations.fits which is the observation to move the wavelength scale to
     
    scale_date : type - string (optional -default '191211')
        date in the form of yymmdd for the scale_star observation
        
    Returns
    -------
    wavelegnth : type - numpy nd-array
        evenly log scaled wavlength grid for the stellar fibres which has a shape = wavelength x order = 22 600 x 40
    
    telluric_spec : type - numpy np-array
        flux for telluric observation which has been weighted averaged over all of the stellar fibres and had an airmass correction applied which has a shape = wavelength x order = 22 600 x 40
    
    error_spec : type - numpy nd-array
        error flux for telluric observation which has been weighted averaged over all of the stellar fibres and had an airmass correction applied which has a shape = wavelength x order = 22 600 x 40
    
    target_star : type - list
        this list contains four elements
        element 0 - string : name of the star in the obs_night observation
        element 1 - byte (string) : observation fits file name as in veloce_observations.fits
        element 2 - byte (int) : date of the observation as in veloce_observations.fits
        element 3 - byte (float) : modified julian date for the observation
    
    telluric_star : type - list
        this list contains four elements
        element 0 - string : name of the telluric star found
        element 1 - byte (string) : telluric fits file name as in veloce_observations.fits
        element 2 - byte (int) : date of the telluric star observation as in veloce_observations.fits
        element 3 - byte (float) : modified julian date for the telluric star observation
    """ 
    # find a telluric star for the given observation
    target_star, telluric_star = find_telluric_star(obs_night, time,date)
    # scrunch the data for both the telluric star and observation onto scale_star wavelength scale with no barycentric correction
    
    if scrunch:
    	wavelength, flux_scale, flux_scale_err, flux_telluric, flux_telluric_err, airmass_telluric = log_scale_interpolation(scale_star,telluric_star[1].decode('utf-8'), scale_date, telluric_star[2].decode('utf-8'), BC = False)
    	wavelength, flux_scale, flux_scale_err, flux_star, flux_star_err, airmass_star = log_scale_interpolation(scale_star,target_star[1].decode('utf-8'), scale_date,target_star[2].decode('utf-8'), BC = False)

    else:
        wavelength, flux_scale, flux_scale_err, flux_telluric, flux_telluric_err, airmass_telluric = log_scale_interpolation(scale_star,telluric_star[1].decode('utf-8'), scale_date, telluric_star[2].decode('utf-8'), BC = False, num_points = 3900)
        wavelength, flux_scale, flux_scale_err, flux_star, flux_star_err, airmass_star = log_scale_interpolation(scale_star,target_star[1].decode('utf-8'), scale_date,target_star[2].decode('utf-8'), BC = False, num_points = 3900)
        

    # create empty lists to fill with template spectrum
    telluric_spec = np.zeros([np.shape(flux_telluric)[0],np.shape(flux_telluric)[1]])
    error_spec = np.zeros([np.shape(flux_telluric)[0],np.shape(flux_telluric)[1]])
    over_error_spec = np.zeros([np.shape(flux_telluric)[0],np.shape(flux_telluric)[1]])
    weights = np.ones([np.shape(flux_telluric)[0],np.shape(flux_telluric)[1]])
	
    
    # iterate through the fibres and orders creating the spectrum
    for fibre in range(np.shape(flux_telluric)[2]):
        print('Working on telluric correction on fibre: ',fibre)
        for order in range(np.shape(flux_telluric)[1]):
            # find the scale factor that the order needs to be shifted by to make flat spectrum
            mask = np.isnan(flux_telluric[:,order,fibre])
            scale = np.median(flux_telluric[:,order,fibre][~mask])
            for i in range(np.shape(flux_telluric)[0]):
                # compute the weighted average
                weight = 1/(flux_telluric_err[i,order,fibre]/scale)**2
                telluric_spec[i,order] += (flux_telluric[i,order,fibre]/scale)*weight
                error_spec[i,order] += flux_telluric_err[i,order,fibre]/scale
                over_error_spec[i,order] += weight
                weights[i,order] += 1
    telluric_spec /= over_error_spec*weights
    over_error_spec /= weights
    error_spec = (1/over_error_spec)**0.5
    
    if B_plus == None:
        B_plus_saved = []
    else:
        B_plus_saved = B_plus    
    for order in range(40):
        print('Working on telluric masking on order: ' + str(order))
        if len(B_plus_saved)==40:
            B_plus = B_plus_saved[order]
        elif scrunch:
            # read in pre-saved matrix
            B_plus = np.load('/priv/avatar/ehold13/B_plus_num_points_22600_order'+str(order)+'.npy', allow_pickle = True)
        telluric_spec[:,order], Bplus = telluric_masking(wavelength[:,order], telluric_spec[:,order], order, Bplus = B_plus)
        #np.save('/priv/avatar/ehold13/B_plus_num_points_22600_order'+str(order)+'.npy', np.array(Bplus),allow_pickle = True)            
        if len(B_plus_saved) != 40:
            B_plus_saved.append(Bplus)
        
    for order in range(40):
        # scale the spectrum and error
        telluric_spec_mask = np.isnan(telluric_spec[:,order])
        scale = np.median(telluric_spec[:,order][~telluric_spec_mask])
        telluric_spec[:,order] /= scale
        error_spec[:,order] /= scale
        for wave in range(np.shape(telluric_spec)[0]):
            # remove 0 points
            if telluric_spec[wave,order]==0:
                telluric_spec[wave,order] = np.nan
                error_spec[wave,order] = np.nan
                 
    
    # apply an airmass correction
    if airmass_corr:          
        telluric_spec = telluric_spec**((airmass_star[1]/airmass_telluric[1]))
                       
    return wavelength, telluric_spec, error_spec, target_star, telluric_star, B_plus_saved

def barycentric_correction(template_obs, star_obs, template_date, star_date, table_dir='/priv/avatar/velocedata/anu_processing/'):
    """
    Description
    -----------
    This code will put a template spectrum and star spectrum onto a common evenly log spaced wavelength grid.
    
    Parameters
    ----------
    template_obs: type - string
        fits file name from veloce_observations.fits that the wavelength scale is being shifted to (e.g. '11dec30096o.fits')
        
    star_obs : type - string
        fits file name from veloce_observations.fits that will have the wavelength scale shifted to template_obs (e.g. '11dec30096o.fits')  
    
    Returns
    -------
    template_delta_lambda.value : type - float
        v/c for template_obs where v is the barycentric velocity
    
    star_delta_lambda.value : type - float
        v/c for star_obs where v is the barycentric velocity
    """
    # find and save the information for the template and for the star obervations which are found in veloce_observations.fits file
    veloce_obs = Table.read(table_dir + 'veloce_observations.fits')
    template_i = 0
    template_j = 0
    template_obs_num = 0
    star_i = 0
    star_j = 0
    star_obs_num = 0
    template_spectrum_dir = ''
    star_spectrum_dir = ''
    for star in veloce_obs:
        template_j = 0
        star_j = 0
        for obs in star[7]:
            if obs.decode("utf-8") == template_obs and veloce_obs[template_i][8][template_j].decode('utf-8') == template_date:
                template_fits_row = veloce_obs[template_i]
                template_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+veloce_obs[template_i][8][template_j].decode("utf-8")+'/'+obs.decode("utf-8")[0:10]+'oi_extf.fits'
                template_obs_num = template_j
                airmass_template = pyfits.open(template_spectrum_dir)[0].header['AIRMASS']
                
            if obs.decode("utf-8") == star_obs and veloce_obs[star_i][8][star_j].decode('utf-8') == star_date:
                star_fits_row = veloce_obs[star_i]
                star_spectrum_dir = '/priv/avatar/velocedata/Data/spec_211202/'+veloce_obs[star_i][8][star_j].decode("utf-8")+'/'+obs.decode("utf-8")[0:10]+'oi_extf.fits'
                star_obs_num = star_j
                airmass_star = pyfits.open(star_spectrum_dir)[0].header['AIRMASS']
                
            template_j += 1
            star_j += 1
        template_i += 1 
        star_i += 1   
    
    # check to see if directories found and sread in the data
    if template_spectrum_dir == '':
        raise UserWarning("Didn't find template!")
    if star_spectrum_dir == '':
        raise UserWarning("Didn't find star!")
        
    template = pyfits.open(template_spectrum_dir)
    star = pyfits.open(star_spectrum_dir)
    
    
    # calculate barycentric velocity correction for the template and for the star (v/c)
    template_BC = get_BC_vel(template_fits_row[6][template_obs_num]+2400000.5, ra = template_fits_row[1][0], dec = template_fits_row[1][1], obsname = 'SSO')
    template_delta_lambda = template_BC[0][0]*u.m*u.s**-1/(c.c.to(u.m/u.s))
            
    star_BC = get_BC_vel(star_fits_row[6][star_obs_num]+2400000.5, ra = star_fits_row[1][0], dec = star_fits_row[1][1], obsname = 'SSO')
    star_delta_lambda = star_BC[0][0]*u.m*u.s**-1/c.c.si

    return template_delta_lambda.value, star_delta_lambda.value
    

def generate_template(file_paths, dates, save_spect = False, save_name = ''):
    """
    Description
    -----------
    This code will create a template spectrum on a evenly spaced log wavelength scale.
    
    Parameters
    ----------
    file_paths : type - list
        each element is a string for a fits files with observations to make into template from veloce_observations.fits, typically these observations will come from observations from a few consecutive nights of observations with good observing conditions
    
    dates : type - list
        each element is the date in the form yymmdd for the corresponding observation in file_paths
    
    save_spect : type - boolean (optional - default False)
        if True will save a fits file of spectrum, wavelength and error to the save_name directory
        
    save_name : type - string (optional - deault '')
        directory to save the fits file
    
    Returns
    -------
    wavelength : type - numpy nd-array
        evenly spaced log wavelength grid which has a shape = wavelength x order = 22 600 x 40
    
    template : type - numpy nd-array
        the tellurically corrected weighted average between observations and fibres template spectrum which has a shape = wavelength x order = 22 600 x 40
    
    error_template : type - numpy nd-array
        error spectrum for the template which has a shape = wavelength x order = 22 600 x 40
     """    
    # scrunch the observation data for first observation to get sizes for lists
    wavelength_standard, flux_t, flux_t_err, flux_s, flux_s_err,airmass = log_scale_interpolation(file_paths[0],file_paths[0],dates[0],dates[0])
    
    # generate a blank spectrum template to fill
    template_spectrum = np.zeros([np.shape(flux_t)[0],np.shape(flux_t)[1],np.shape(flux_t)[2]], dtype = object)
    template_spectrum_error = np.zeros([np.shape(flux_t)[0],np.shape(flux_t)[1],np.shape(flux_t)[2]], dtype = object)
    template_spectrum_one_on_error = np.zeros([np.shape(flux_t)[0],np.shape(flux_t)[1],np.shape(flux_t)[2]], dtype = object)
   
    # iterate through each observation that is wanted in the template
    num_good_pixels = np.ones([np.shape(flux_t)[0],np.shape(flux_t)[1],np.shape(flux_t)[2]],dtype = object)
    B_plus_saved = None
    
    for index,obs in enumerate(file_paths):
        print('Generating template for observation: ', str(index+1)+'/'+str(len(file_paths)))
        # find airmass corrected telluric corrections from before and after observation
        wave_tell_b, telluric_spec_b, telluric_err_spec_b, target_info_b, telluric_info_b, B_plus_saved = telluric_correction(obs,'before',dates[index], B_plus = B_plus_saved)
       
        wave_tell_a, telluric_spec_a, telluric_err_spec_a, target_info_a, telluric_info_a, B_plus_saved = telluric_correction(obs, 'after',dates[index], B_plus = B_plus_saved)
        # take the time weighted average of the before and after telluric spectra, if there was only one, then no need for this step
        if telluric_info_a[1]!= telluric_info_b[1]:
            telluric_spec = (telluric_spec_a*(target_info_b[3] - telluric_info_b[3]) + telluric_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3])
            # propagate the errors, a + b = (sig_a**2 + sig_b**2)**0.5
            telluric_err_spec = (((telluric_err_spec_a*(target_info_b[3] - telluric_info_b[3]))/(telluric_info_a[3]-telluric_info_b[3]))**2 + ((telluric_err_spec_b*(target_info_a[3] - telluric_info_a[3]))/(telluric_info_a[3]-telluric_info_b[3]))**2)**0.5
        else:
            telluric_spec = telluric_spec_a
            telluric_err_spec = telluric_err_spec_a
        # scrunch the observation data
        wavelength, flux_t, flux_t_err, flux_s, flux_s_err, airmass = log_scale_interpolation(file_paths[0],obs,dates[0], dates[index],BC=False)
        
        
        # for each fibre in the observation, divide by the telluric spectrum
        for fibre in range(19):
            for order in range(40):
                for wave in range(np.shape(flux_s)[0]):
                    # propagate the errors a/b = ([sig_a/a]**2 + [sig_b/b]**2)**0.5
                    flux_s_err[wave,order,fibre] = ((flux_s_err[wave,order,fibre]/flux_s[wave,order,fibre])**2 + (telluric_err_spec[wave,order]/telluric_spec[wave,order])**2)**0.5
            flux_s[:,:,fibre] /= telluric_spec
            flux_s_err *= flux_s
            
        
        # calculate the barycentric correction for the observation
        template_delta_lambda, star_delta_lambda = barycentric_correction(file_paths[0],obs, dates[0],dates[index])   
         
        # apply the barycentric correction
        wavelength += star_delta_lambda*wavelength
        
        # define numpy nd-arrays to store the data in
        good_pixels = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1],np.shape(flux_s)[2]],dtype = object)
        error_good_pixels = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1],np.shape(flux_s)[2]],dtype = object)
        one_over_error_good_pixels = np.zeros([np.shape(flux_s)[0],np.shape(flux_s)[1],np.shape(flux_s)[2]],dtype = object)
        
        # iterate through the fibres
        for fibre in range(19):
            # iterate through the orders
            for order in range(40):
                # want to add all observations together, so need them on the same wavelength scale
                obs_interp_func = InterpolatedUnivariateSpline(wavelength[:,order], flux_s[:,order,fibre], k =1)
                obs_err_interp_func = InterpolatedUnivariateSpline(wavelength[:,order], flux_s_err[:,order,fibre], k = 1)
        
                flux_s[:,order,fibre] = obs_interp_func(wavelength_standard[:,order])
                flux_s_err[:,order,fibre] = obs_err_interp_func(wavelength_standard[:,order])
                
                # pull out spectrum
                spectrum = flux_s[:,order,fibre]
                # find scale for spectrum and divide the spectrum and the spectrum error by this scale 
                mask = np.isnan(spectrum)
                scale = np.median(spectrum[~mask])
                spectrum /= scale
                error = flux_s_err[:,order,fibre]/scale
                       
                
                # find the weighted average for the spectrum
                for i in range(len(spectrum)):
                    good_pixels[i, order, fibre] = spectrum[i]/error[i]**2
                    error_good_pixels[i,order,fibre] = error[i]
                    one_over_error_good_pixels[i,order,fibre] = 1/error[i]**2
                    num_good_pixels[i,order,fibre] += 1

        # add the observation to the list and do the same for the errors
        template_spectrum += good_pixels
        template_spectrum_error += error_good_pixels
        template_spectrum_one_on_error += one_over_error_good_pixels

       
    # divide by the number of pixels that went into each sum, and divide by the weights (1/sum(error))**0.5
    template_spectrum_error /= num_good_pixels
    template_spectrum_one_on_error /= num_good_pixels
    template_spectrum /=num_good_pixels*template_spectrum_one_on_error
    template_spectrum_error = (1/template_spectrum_one_on_error)**0.5
    
    # find the median fibre height to check for bad pixels, only include the stellar fibres
    median_fibre_spectrum = np.median(template_spectrum,2)
       
    # for each fibre, find the median difference between it and median_fibre_spectrum and remove pixels with a difference much larger than this
    template = np.zeros([np.shape(template_spectrum)[0],np.shape(template_spectrum)[1]])
    error = np.zeros([np.shape(template_spectrum)[0],np.shape(template_spectrum)[1]])
    one_on_error = np.zeros([np.shape(template_spectrum)[0],np.shape(template_spectrum)[1]])
    weights = np.ones([np.shape(template_spectrum)[0],np.shape(template_spectrum)[1]])
    
    for fibre in range(19):
        diff = [abs(template_spectrum[:,:,fibre] - median_fibre_spectrum[:,:])]
        med_diff = np.median(diff)
        # the distance from the median difference for each point
        diff = abs(diff - med_diff)[0,:,:]
        
        for order in range(np.shape(diff)[1]):
            scale = np.median(template_spectrum[:,order,fibre])
            
            for wave in range(np.shape(diff)[0]):
                # only include a value if ratio is less than 2 of the median value
                if diff[wave,order]/med_diff<=2:
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
    print('Doing a signal to noise cut.')
    for order in range(40):
        for wave in range(np.shape(diff)[0]):
            if (template[wave,order]==0) | (template[wave,order]<3*error[wave,order]):
                template[wave,order] = np.nan
                error[wave,order] = np.nan
                
    # return the template spectrum with weighted average
    if save_spect:
        primary_hdu = pyfits.PrimaryHDU(template)
        image_hdu = pyfits.ImageHDU(wavelength_standard)
        image_hdu2 = pyfits.ImageHDU(error)
        hdul = pyfits.HDUList([primary_hdu, image_hdu, image_hdu2])
        hdul.writeto('/home/ehold13/veloce_scripts/'+save_name+'.fits') 
    return wavelength_standard, template, error
    
