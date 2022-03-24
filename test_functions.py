from astropy.table import Table
from rv import log_scale_interpolation
from rv import find_telluric_star
from rv import telluric_correction

#---------------------
#TESTING OF FUNCTIONS
#---------------------

#________________________
# log_scale_interpolation
#________________________  

#w,t,terr,s,serr,airmass = log_scale_interpolation('11dec30096o.fits','11dec30096o.fits')

#____________________
# find_telluric_star
#____________________    
#st, ts = find_telluric_star('15dec30098o.fits','closest')

#_____________________
# telluric_correction
#_____________________
#dd = Table.read('veloce_observations.fits')
#w_t, s_t, e_t = telluric_correction('15dec30098o.fits','closest')

#________________________
# barycentric_correction
#________________________

#a,b = barycentric_correction('11dec30096o.fits', '11dec30097o.fits')

#___________________
# generate_template
#___________________

# testing_temp_files = ['11dec30096o.fits', '11dec30097o.fits', '12dec30132o.fits', '12dec30133o.fits', '12dec30134o.fits', '13dec30076o.fits', '13dec30077o.fits', '14dec30066o.fits', '14dec30067o.fits', '14dec30068o.fits', '15dec30097o.fits', '15dec30098o.fits', '15dec30099o.fits']

#w, s, e = generate_template(testing_temp_files)

#primary_hdu = pyfits.PrimaryHDU(s)
#image_hdu = pyfits.ImageHDU(w)
#image_hdu2 = pyfits.ImageHDU(e)
#hdul = pyfits.HDUList([primary_hdu, image_hdu, image_hdu2])
#hdul.writeto('Tau_Ceti_Template_dec2019_tellcor_1.fits')

#_______________________
# how to save fits file
#_______________________
#primary_hdu = pyfits.PrimaryHDU(s_t)
#image_hdu = pyfits.ImageHDU(w_t)
#image_hdu2 = pyfits.ImageHDU(e_t)
#hdul = pyfits.HDUList([primary_hdu, image_hdu, image_hdu2])
#hdul.writeto('telluric_13dec2019.fits')


