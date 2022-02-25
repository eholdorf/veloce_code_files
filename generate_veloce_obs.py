import numpy as np
import glob as glob
import astropy.io.fits as pyfits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
#from astroquery.gaia import Gaia
import astropy.units as u
import re
import get_observations as g_o


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

Teff = [0]*len(star_names)
K_pl = [0]*len(star_names)
s_n = list(star_names)
obs_current_new = Table.read('Veloce_Observing_Status_Current_Targets_NEW.csv')
for line in obs_current_new[2:-1]:
    for name in star_names:
        if str(line[1]) in name:
            Teff[s_n.index(name)] = int(line[24])           
obs_current_old = Table.read('Veloce_Observing_Status_Current_Targets_OLD.csv')
for line in obs_current_old[2:-1]:
    for name in star_names:
        if str(line[1]) in name:
            if Teff[s_n.index(name)] == 0:
                Teff[s_n.index(name)] = int(line[24])
            
obs_current_old = Table.read('Veloce_Observing_Status_Aug21_Oct21_planning.csv')
for line in obs_current_old[2:-1]:
    for name in star_names:
        if str(line[1]) in name:
            if Teff[s_n.index(name)] == 0:
                Teff[s_n.index(name)] = int(line[19]) 
            if K_pl[s_n.index(name)] == 0:
                K_pl[s_n.index(name)] = float(line[5])
                
obs_current_old = Table.read('Veloce_Observing_Status_Mar21_planning.csv')
for line in obs_current_old[2:-1]:
    for name in star_names:
        if str(line[1]) in name:
            if Teff[s_n.index(name)] == 0:
                Teff[s_n.index(name)] = int(line[19])  
            if K_pl[s_n.index(name)] == 0:
                K_pl[s_n.index(name)] = float(line[5])         
# get name, TOI status and colour out of text file
toi_info = open('TOIs.txt')
toi_info_ = []
for line in toi_info:
    name = re.search('TOI [0-9]*',line)
    status = re.search(', [A-Z]*',line)
    colour = re.search(', [0-9,.,-][0-9,.,-]*',line)
    toi_info_.append((name.group(0),status.group(0)[1:],colour.group(0)[1:]))

toi_info.close()

# making a plot of the TOI's
toi_index = []
count = 0
for obs in obs_type:
    if obs == 'TOI':
        toi_index.append(count)
    count += 1
tois = [star_names[i] for i in toi_index]
#print(tois)      
toi_radecs = [uniq_radec[i] for i in toi_index]

toi_ra = [radec[0] for radec in toi_radecs]
toi_dec = [radec[1] for radec in toi_radecs]

toi_size = []
for i in toi_index:
    toi_size.append(num_obs[i])
    
# get colour out for plot
bp_rps = []

bp_rp = np.array([float(toi[2]) for toi in toi_info_])
mask = bp_rp>0.6
bp_rp = bp_rp[mask]
toi_ra = np.array(toi_ra)[mask]
toi_dec = np.array(toi_dec)[mask]
toi_size = np.array(toi_size)[mask]

plt.figure()
plt.scatter(toi_ra[6],toi_dec[6], s=200, c= bp_rp[6],cmap = 'Greys', label = '200')
plt.scatter(toi_ra[50],toi_dec[50], s=toi_size[50], c= bp_rp[50],cmap = 'Greys', label = str(toi_size[50]))
plt.scatter(toi_ra[51],toi_dec[51], s=toi_size[51], c= bp_rp[51],cmap = 'Greys', label = str(toi_size[51]))
plt.scatter(toi_ra[10],toi_dec[10], s=toi_size[10], c= bp_rp[10],cmap = 'Greys',label = str(toi_size[10]))
#plt.scatter(toi_ra[-1],toi_dec[-1], s=toi_size[-1], c= bp_rp[-1],cmap = 'Greys',label = str(toi_size[-1]))
plt.scatter(toi_ra,toi_dec,s=toi_size, c = bp_rp,cmap = 'RdYlBu_r')
plt.xlabel('RA (deg)')
plt.ylabel('dec (deg)')
plt.colorbar(label =r'$B_p - R_p$')
plt.legend(loc = 'best')
plt.show()

for name in tois:
    for toi in toi_info_:
        if str(toi[0][4:]) in str(name):
            i = toi_index[tois.index(name)]
            obs_type[i] += toi[1]
#find directory of each data set
directories = []
count = 0
for star in fits:
    directory = g_o.get_folder(star)
    directories.append(directory)
    print(count)
    count+=1
print(directories)
t = Table([star_names, uniq_radec,obs_type,num_obs,Teff,K_pl, jds, fits,directories], names = ('star_names','ra_dec','obs_type','number_obs','T_eff','K_pl','julian_obs_dates', 'fits_names','directory'))
t.write('veloce_observations.fits', format = 'fits')



