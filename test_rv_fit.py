from rv.least_squares_rv import *
from rv.main_funcs import barycentric_correction
from astropy.table import Table
from os.path import exists

save_dir = './' #'/home/ehold13/veloce_scripts/obs_corrected_fits/'
data_logs = Table.read('/home/ehold13/veloce_scripts/veloce_observations.fits')
Tau_Ceti = data_logs[data_logs['star_names']=='10700'][0]

files = ['11dec30096o.fits','11dec30097o.fits','12dec30132o.fits','12dec30133o.fits','12dec30134o.fits', '13dec30076o.fits', '13dec30077o.fits', '14dec30066o.fits', '14dec30067o.fits', '14dec30068o.fits', '15dec30097o.fits', '15dec30098o.fits', '15dec30099o.fits']

#files.extend([Tau_Ceti[7][i].decode('utf-8') for i in range(len(Tau_Ceti))])

file_dates = ['191211','191211','191212','191212','191212','191213','191213','191214','191214','191214','191215','191215','191215']

#file_dates.extend([Tau_Ceti[8][i].decode('utf-8') for i in range(len(Tau_Ceti))])

for i,file_name in enumerate(files):
    if not exists('/priv/avatar/ehold13/obs_corrected/'+file_name[0:10]+'_corrected.fits'):
        print('Working on fits: '+ str(file_name))
        s,w,se = create_observation_fits('11dec30096o.fits',file_name,file_dates[i],'/priv/avatar/ehold13/obs_corrected/')
#s,w,se = create_observation_fits('11dec30096o.fits','11dec30088o.fits','191211','/priv/avatar/ehold13/obs_corrected/')

# do some testing on files which weren't used to make the template
fitting_files = [f[0:10] for f in files]
#fitting_files = fitting_files[:1] #!!!

        
#FIXME: There should be a better way than this of dealing with "bad" orders.
orders= list(range(11,15))
orders.extend(list(range(17,39))) #!!! Was 40
#To do all orders... (note issues with NaNs)
orders= list(range(5,5))
orders.extend(list(range(6,10)))
orders.extend(list(range(11,15))) 
orders.extend(list(range(17,39)))

orders = [6,7,13,14,17,25,26,27,28,30,31,33,34,35,36,37] #!!! Temp, orders which have low telluric contamination based on Bplus matrix size

velocity_err = np.zeros([len(fitting_files),len(orders)])
velocity = np.zeros([len(fitting_files),len(orders)])
velocity_uncertainty = np.zeros([len(fitting_files),len(orders)])

Tau_Ceti_Template = pyfits.open('/home/ehold13/veloce_scripts/Tau_Ceti_Template_dec2019_telluric_patched_4.fits')


mjds = []
for file_ind, fit in enumerate(fitting_files):
    print('Calculating RV for: ' + str(fit)+'...')
    obs_file_path = '/priv/avatar/ehold13/obs_corrected/'+fit+'_corrected.fits'
    obs = pyfits.open(obs_file_path)
    procfile = '/priv/avatar/velocedata/Data/spec_211202/1912' + fit[:2] + '/' + fit + 'oi_extf.fits'
    hh = pyfits.getheader(procfile)
    mjds.append(hh['UTMJD'])

    spect = obs[0].data
    wavelength = obs[1].data
    spect_err = obs[2].data
        
    if False:
        order = 36
        mask = np.isnan(Tau_Ceti_Template[0].data[:,order])  
        temp_func = InterpolatedUnivariateSpline(Tau_Ceti_Template[1].data[:,order][~mask], Tau_Ceti_Template[0].data[:,order][~mask], k=1)
        
        obs_mask = np.isnan(spect[401:3601,order,0])
        sp = rv_fitting_eqn_old([-24,1e-3,1e-1,1e-1],wavelength[401:3601,order,0][~obs_mask],spect[401:3601,order,0][~obs_mask],spect_err[401:3601,order,0][~obs_mask],temp_func, return_spec = True)

        plt.figure()
        plt.plot(wavelength[401:3601,order,0][~obs_mask],spect[401:3601,order,0][~obs_mask], label = 'Original Spectrum')
        plt.plot(wavelength[401:3601,order,0][~obs_mask],sp, label = 'Fitted Spectrum')
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Flux')
        plt.title(fit)
        plt.legend(loc='best')
        #plt.ylim(0,1.2)
        plt.show()
    for order_ind, i in enumerate(orders):
        #FIXME: The template can not have NaNs in it! Just make it smooth over gaps.

        if np.sum(np.isnan(Tau_Ceti_Template[0].data[830:3200,i]) > 0):
            raise UserWarning("Can not have NaNs in template!")
        temp_wave = Tau_Ceti_Template[1].data[:,i]
        temp_spec = Tau_Ceti_Template[0].data[:,i] 
        
        
        
        #FIXME: The template shouldn't have to be convolved after the fact. There are some high frequency parts of the 
        #spectrum - where do they come from?
        gg = np.exp(-np.linspace(-2,2,15)**2/2) #Must have an odd length.
        gg /= np.sum(gg)
        temp_spec = np.convolve(temp_spec,gg, mode='same')
        #FIXME end...        
        
        temp_func = InterpolatedUnivariateSpline(temp_wave, temp_spec, k=1) 
        temp_lwave = np.log(temp_wave)
        
        temp_dlwave = temp_lwave[1]-temp_lwave[0]
        rvs = []
        rv_errs = []
        mses = []
        for j in range(19):
            spect_mask = np.isnan(spect[830:3200,i,j])
            
            spect_wave = wavelength[830:3200,i,j][~spect_mask].astype(np.float64) #Converting to float64 is essential!
            spect_spec = spect[830:3200,i,j][~spect_mask]
            
            spect_err_ = np.sqrt(np.abs(spect_err[830:3200,i,j][~spect_mask])) #??? It seems that the errors are too small.
            #if np.sum(np.isnan(spect_err_))>0:
            #    import pdb; pdb.set_trace()
            spect_lwave = np.log(spect_wave)
            
            
            
            # changing intial conditions changes answer and MSE 
            #a = optimise.leastsq(rv_fitting_eqn_old,x0 = [-24,1e-3,1e-3,1e-3], args=(spect_wave, spect_spec, spect_err_, temp_func), epsfcn=1e-6, full_output = True, ftol=1e-6, gtol=1e-6)
            
#rv_fitting_eqn(params, lwave, spect, spect_err, template, lwave0, dlwave, return_spec = False):
            #a = optimise.leastsq(rv_fitting_eqn,x0 = [-25,1e-3,1e-3,1e-3], args=(spect_lwave, spect_spec, spect_err_, temp_spec, temp_lwave[0], temp_dlwave), \
            #   epsfcn=1e-4, full_output = True, ftol=1e-7, gtol=1e-7)     
            if 0==len(spect_lwave):
                continue
                
            a = optimise.least_squares(rv_fitting_eqn,x0 = [-26,0,0,0], args=(spect_lwave, spect_spec, spect_err_, temp_spec, temp_lwave[0], temp_dlwave), \
                jac=rv_jac, method='lm') 

            print(a.x)
            print(a.status)
            cov = np.linalg.inv(np.dot(a.jac.T,a.jac))

            if False:
                plt.figure()
                plt.plot(spect_wave,a[2]['fvec'])
                plt.show()
            #import pdb; pdb.set_trace()

            #!!! To be neatened. All velocities should be saved in a 2D array !!!
            if a.success: #a[0][0] != -24 and not (a[1] is None):
               rvs.append(a.x[0])
               mse = np.mean(a.fun**2) #np.mean(a[2]['fvec']**2)
               mses.append(mse)
               #Multiply the error but the square root of chi-squared. !!! Why is chi-squared so high? !!!
               rv_errs.append(np.sqrt(mse)*np.sqrt(cov[0,0]))
               #Now make a plot to see if the minimum is unique!
               if False:
                   rv_test = np.linspace(-24.1,-23.9,201)
                   this_chi2 = test_rv_chi2(a.x, rv_test, spect_lwave, spect_spec, spect_err_, temp_spec, temp_lwave[0], temp_dlwave)
                   plt.figure(j)
                   #plt.clf()
                   plt.plot(rv_test, this_chi2)

        rvs = np.array(rvs)
        rv_errs = np.array(rv_errs)
        weights = 1/rv_errs**2
        wtmn_rv = np.sum(weights*rvs)/np.sum(weights)
        wtmn_rv_err = 1/np.sqrt(np.sum(weights))
        print("Weighted mean RV (km/s): {:.4f} +/- {:.4f}".format(wtmn_rv, wtmn_rv_err))
        velocity[file_ind, order_ind] = wtmn_rv
        velocity_uncertainty[file_ind, order_ind] = wtmn_rv_err
        #The mean square error will be significantly larger than 1 (e.g. more than 1.5) if fiber to fiber 
        #variations are determined by more than just random errors.
        print("MSE over fibers: {:.2f}".format(np.mean((wtmn_rv - rvs)**2/rv_errs**2)))
        print("MSE sum for each fiber: {:.4f}".format(np.sum(mses)))
        if False:
            plt.figure()
            plt.errorbar(np.arange(len(rvs))+1, rvs, rv_errs, fmt='.')
            plt.title('Order:' + str(i))
            plt.xlabel('Fiber num')
            plt.ylabel('Velocity (km/s)')
            plt.show()
        
        obs_day = obs_file_path[35:37]
        BC_t, BC_star = barycentric_correction('11dec30096o.fits',obs_file_path[35:45]+'o.fits','191211','1912'+obs_day)
        
        print('True Velocity (km/s) ', BC_star*c.c.to(u.km/u.s).value)
        print('Velocity Difference (m/s) ', wtmn_rv*1000 - BC_star*c.c.to(u.m/u.s).value)
        
        if abs(wtmn_rv*1000 - BC_star*c.c.to(u.m/u.s).value) > 100:
            print(wtmn_rv*1000 - BC_star*c.c.to(u.m/u.s).value)
            print(wtmn_rv*1000)
            print(BC_star*c.c.to(u.m/u.s).value)
            print(fit, i)
            #raise UserWarning("High Error!!")

        print('Percentage Error ', 100*(wtmn_rv - BC_star*c.c.to(u.km/u.s).value)/(BC_star*c.c.to(u.km/u.s).value))
        velocity_err[file_ind, order_ind] = (wtmn_rv*1000 - BC_star*c.c.to(u.m/u.s).value)

if False:
    for order in range(len(orders)):        
        plt.figure()   
        plt.plot(velocity_err[:,order])
        plt.show()

print('Velocity uncertainty, list of orders (m/s): {:.1f}'.format(np.std(np.mean(velocity_err, axis=1)))) # was 24:34
print('Internal dispersion, based on scatter between orders (m/s): ')
simple_std = np.std(velocity_err, axis=1)/np.sqrt(len(orders))
print(simple_std)
simple_means = np.mean(velocity_err, axis=1)
for i in range(len(simple_means)):
    print("{:.6f},{:.1f},{:.1f}".format(mjds[i], simple_means[i], simple_std[i]))

plt.figure()
plt.imshow(velocity_err)
plt.xlabel('Order')
plt.ylabel('Observation')
plt.colorbar(label = 'RV Error (m/s)')
plt.xticks(list(range(len(orders))),orders)
# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12],fitting_files)
plt.show()
if False:
    plt.figure()
    plt.errorbar([11,11,12,12,12,13,13,14,14,14,15,15,15,17,17,19,19,19,19,19,21,28,28,28,28,28],np.median(velocity_err[:,:], 1),yerr=np.sum(velocity_uncertainty,1)*1000/len(velocity_uncertainty[:,0]),fmt = 'k.')
    plt.xlabel('Date')
    plt.ylabel('Median Velocity Error')
    words = [str(i)+ 'dec19' for i in [11,12,13,14,15,17,19,21,28]]
    plt.xticks([11,12,13,14,15,17,19,21,28], words)
    plt.show()


