import numpy as np
import glob as glob
import astropy.io.fits as pyfits
from astropy.table import Table

def get_fits_path(fits,extension = 'extf'):
    """
    Description
    -----------
    This function will return the file path to the fits files given in the veloce_observations.fits fits_files column for the three different extensions.
    
    Parameters
    ----------
    fits : type - list of strings
        list of fits file paths wanted from the veloce_observations.fits file
    
    extension : type - string (optional, defult 'extf')
        type of file wanted, 'extf', 'extv' or 'exts'
    
    Returns
    -------
    all_files : type - list of strings
        list containing list of all of the desired file paths
    """
    all_files = []
    for fit in fits:
        # decode the byte to a string and extract the start of the string which contains the date and observation
        fit = fit.decode("utf-8")[0:10]
        # check the given extension is valid
        if extension not in ['extf','extv','exts']:
            print("not a valid extension, choose from: 'extf', 'extv' or 'exts'")
            return
        # extract the path to the file
        desired_file = glob.glob('/priv/avatar/velocedata/Data/spec_211202/[12]?????/'+fit+'oi_'+extension+'.fits')
        # add the path to the list of file paths
        all_files.append(desired_file[0])
    print(all_files)
    return all_files

#testing
#dd = Table.read('veloce_observations.fits')        
#get_fits_path([dd[0][7][0],dd[0][7][1]])
        
        

