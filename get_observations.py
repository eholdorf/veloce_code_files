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
    # initiate list length of all of the files
    all_files = ['']*len(fits)
    i = 0
    for fit in fits:
        # decode the byte to a string and extract the start of the string which contains the date and observation
        fit = fit.decode("utf-8")[0:10]
        # check the given extension is valid
        if extension not in ['extf','extv','exts']:
            print("not a valid extension, choose from: 'extf', 'extfv' or 'extfs'")
            return
        # extract the path to the file
        desired_file = glob.glob('/priv/avatar/velocedata/Data/spec_211202/[12]?????/'+fit+'oi_'+extension+'.fits')
        # add the path to the list of file paths
        all_files[i] = desired_file[0]
        i += 1
    print(all_files)
    return all_files

#testing
#dd = Table.read('veloce_observations.fits')        
#get_fits_path([dd[0][7][0],dd[0][7][1]])
        
def get_folder(fits):
    all_folders = ['']*len(fits)
    i = 0
    for fit in fits:
    # decode the byte to a string and extract the start of the string which contains the date and observation
        if fit != '':
            fit = fit[0:10]    

            # extract the path to the file
            desired_file = glob.glob('/priv/avatar/velocedata/Data/Raw/[12]?????/ccd_3/'+fit+'.fits')
            if len(desired_file)==0:
                desired_file = glob.glob('/priv/avatar/velocedata/Data/spec_211202/[12]?????/'+fit+'oi_extf.fits')
                if len(desired_file)==0:
                    folder = 'DNF'
                else:
                    folder = desired_file[0][41:47]
            else:
                folder = desired_file[0][33:39]
            all_folders[i] = folder
            i += 1
    #print(all_folders)
    return all_folders
    
#testing
#dd = Table.read('veloce_observations.fits')        
#get_folder([dd[0][7][0],dd[0][7][1]])
