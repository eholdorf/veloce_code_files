import numpy as np
import glob as glob
import astropy.io.fits as pyfits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.units as u


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

obs_type = []
while len(obs_type)<len(fits):
    obs_type.append('TARGET')
# not found in list: 108249    
BSTARS = ['10144', '14228', '37795', '47670', '50013', '56139', '89080', '91465', '93030', '98718', '105435', '105937', '106490', '108248', '108483', '109026', '109668', '110879', '118716', '120324', '121263', '121743', '121790', '122451', '125238', '127972', '129116', '132058', '134481', '136298', '136504', '138690', '139365', '142669', '143018', '143118', '143275', '144470', '157246', '158094', '158427', '158926', '160578', '165024', '169022', '175191', '209952', 'HR4468'] 

STABLE_STARS = ['10700', '85512', '190248', 'Gl87', '194640', '144628']

# not found in list: 211.01
BINARY = ['AST303', '123.01', '129.01', '217.01', '544.01']

#not found in list: 166.01
REMOVED = ['151.01',]

NEB = ['379.01', '399.01', '419.01', '803.01']

ROTATOR = ['523.01']

# changing the obsservation type for each star based on logs
for name in star_names:
    if '.0' in name:
        obs_type[star_names.index(name)] = 'TOI'

for star in BSTARS:
    for name in star_names:
        if star in name:
            star = name
    obs_type[star_names.index(star)] = 'BSTAR'
    
for star in STABLE_STARS:
    for name in star_names:
        if star in name:
            star = name
    obs_type[star_names.index(star)] = 'STABLE'  
    
for star in BINARY:
    for name in star_names:
        if star in name:
            star = name
    obs_type[star_names.index(star)] = 'BINARY'
    
for star in REMOVED:
    for name in star_names:
        if star in name:
            star = name
    obs_type[star_names.index(star)] = 'REMOVED'  
    
for star in NEB:
    for name in star_names:
        if star in name:
            star = name
    obs_type[star_names.index(star)] = 'NEB' 
    
for star in ROTATOR:
    for name in star_names:
        if star in name:
            star = name
    obs_type[star_names.index(star)] = 'ROTATOR'  
            
uniq_radec = np.array(uniq_radec)
star_names = np.array(star_names)
star_ix = np.array(star_ix)
obs_count = np.unique(star_ix, return_counts=True)[1]
nflat = obs_count[0]
obs_count = obs_count[1:]

#t = Table([star_names, uniq_radec,obs_type,num_obs, jds, fits], names = ('star_names','ra_dec','obs_type','number_obs','julian_obs_dates', 'fits_names'))
#t.write('veloce_observations.fits', format = 'fits')

# making a plot of the TOI's
toi_index = []
count = 0
for obs in obs_type:
    if obs == 'TOI':
        toi_index.append(count)
    count += 1
      
toi_radecs = [uniq_radec[i] for i in toi_index]

toi_ra = [radec[0] for radec in toi_radecs]
toi_dec = [radec[1] for radec in toi_radecs]

toi_size = []
for i in toi_index:
    toi_size.append(num_obs[i])
bp_rps = []
for item in toi_radecs:    
    coord = SkyCoord(ra=item[0], dec=item[1], unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(0.1, u.deg)
    height = u.Quantity(0.1, u.deg)
    r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
    bp_rps.append(r[0]['bp_rp'])
    
print(bp_rps)

plt.figure()
plt.scatter(toi_ra,toi_dec,s=toi_size,c=bp_rps,cmap='RdYlBu_r')
plt.xlabel('RA')
plt.ylabel('dec')
plt.colorbar()
plt.show()
