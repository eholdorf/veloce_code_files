import sys
sys.path.append('/home/ehold13/veloce_scripts/')
from rv.least_squares_rv import *
from rv.main_funcs import barycentric_correction
from astropy.table import Table
import astropy.io.fits as pyfits

# put star name and date of observations here
star_name = '10700'
date = '191228'
template = pyfits.open('/home/ehold13/veloce_scripts/Tau_Ceti_Template_dec2019_telluric_patched_4.fits')

velocity_errors, files, orders = generate_rvs(star_name,date,template)

plt.figure()
plt.imshow(velocity_errors)
plt.xlabel('Order')
plt.ylabel('Observation')
plt.colorbar(label = 'RV Error (m/s)')
plt.xticks(list(range(len(orders))),orders)
plt.yticks(list(range(len(files))),files)
plt.show()
