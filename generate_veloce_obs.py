import numpy as np
import glob as glob
import astropy.io.fits as pyfits
from astropy.table import Table

objfns = glob.glob('oversc_logs/[12]?????/OBJECT.lis')

dirs = []
fits = []
jds = []
radecs = []
names = []

#A n_fits list of star indices
star_ix = []

#The unique RA and Decs of these stars
uniq_radec = []
#The unique names of these stars. As time goes on, we over-write initial names
#to conform with the most recent naming convention.
star_names = []

for fn in objfns:
    dirname = fn[fn.find('/')+1:fn.rfind('/')]
    with open(fn) as file:
        for line in file.readlines():
            #We're not using ''.split() because the name can have spaces in it!
            args = line.split()
            name = ' '.join(args[4:])
            names += [name]
            radec = [float(args[2]), float(args[3])]
            radecs += [radec]
            
            #Firstly, check for mis-labelled flat fields.
            if name[:4] == 'Flat':
                star_ix += [-1]
            else:
                found_star=False
                for ix, star_radec in enumerate(uniq_radec):
                    dist2 = (star_radec[1]-radec[1])**2 + \
                        np.cos(np.radians(star_radec[1]))**2*(star_radec[0]-radec[0])**2
                    if dist2 < 0.017**2: #Within 1 arcmin. Strangely this is needed!
                        star_names[ix] = name
                        found_star=True
                        star_ix += [ix]
                        break

                #If we haven't found the star yet, add a new one!
                if not found_star:
                    uniq_radec.append(radec)
                    star_names.append(name)
                    fits.append([args[0]])
                    jds.append([float(args[1])])
                # if we have found the star, then add the fits file and jd to the list for that star
                if found_star:
                    index = star_names.index(name)
                    fits[index].append(args[0])
                    jds[index].append(float(args[1]))
                        
# number of observations for each star                       
num_obs = [len(fits[i]) for i in range(len(fits))]
for fit in fits:
    diff = abs(max(num_obs) - len(fit))
    while diff > 0:
        fit.append('')
        diff -= 1
        
for jd in jds:
    diff = abs(max(num_obs) - len(jd))
    while diff > 0:
        jd.append(np.NaN)
        diff -= 1
                    
uniq_radec = np.array(uniq_radec)
star_names = np.array(star_names)
star_ix = np.array(star_ix)
obs_count = np.unique(star_ix, return_counts=True)[1]
nflat = obs_count[0]
obs_count = obs_count[1:]

t = Table([star_names, uniq_radec,num_obs, jds, fits], names = ('star_names','ra_dec','number_obs','julian_obs_dates', 'fits_names'))
t.write('veloce_observations.fits', format = 'fits')
