import sys
sys.path.append("/home/ehold13/veloce_scripts/")
from rv.least_squares_rv import *
from rv.main_funcs import barycentric_correction
from astropy.table import Table
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

# put star name, date and template here
star_name ='10700'
date ='191228'
template ='/home/ehold13/veloce_scripts/Tau_Ceti_Template_dec2019_telluric_patched_4.fits'
velocity_errors, files, orders = generate_rvs(star_name,date,template)

if False:
    plt.figure()
    plt.imshow(velocity_errors)
    plt.xlabel("Order")
    plt.ylabel("Observation")
    plt.colorbar(label = "RV Error (m/s)")
    plt.xticks(list(range(len(orders))),orders)
    plt.yticks(list(range(len(files))),files)
    plt.show()
