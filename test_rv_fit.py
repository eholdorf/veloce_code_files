from rv.least_squares_rv import *

save_dir = './' #'/home/ehold13/veloce_scripts/obs_corrected_fits/'
#files = ['11dec30097o.fits', '12dec30132o.fits', '12dec30133o.fits', '12dec30134o.fits', '13dec30076o.fits', '13dec30077o.fits', '14dec30066o.fits', '14dec30067o.fits', '15dec30097o.fits', '15dec30098o.fits', '15dec30099o.fits']
#for file_name in files:
    #s,w,se = create_observation_fits('11dec30096o.fits',file_name,'/home/ehold13/veloce_scripts/obs_corrected_fits/')
#s,w,se = create_observation_fits('11dec30096o.fits','14dec30068o.fits','/home/ehold13/veloce_scripts/obs_corrected_fits/')
fitting_files = ['11dec30096','11dec30097','12dec30132','12dec30133','12dec30134', '13dec30076','13dec30077','14dec30066','14dec30067','14dec30068','15dec30097', '15dec30098', '15dec30099']
fitting_files = fitting_files[:2] #!!!
velocity_err = np.zeros([len(fitting_files),36])
file_ind = 0
for fit in fitting_files:
    Tau_Ceti_Template = pyfits.open('/home/ehold13/veloce_scripts/Tau_Ceti_Template_14dec2019_telluric_patched.fits')
    
    obs_file_path = '/home/ehold13/veloce_scripts/obs_corrected_fits/'+fit+'_corrected.fits'
    obs = pyfits.open(obs_file_path)

    spect = obs[0].data
    wavelength = obs[1].data
    spect_err = obs[2].data
    
        
    if False:
        order = 13
          
        temp_func = InterpolatedUnivariateSpline(Tau_Ceti_Template[1].data[:,order], Tau_Ceti_Template[0].data[:,order], k=1)
        sp = rv_fitting_eqn([-24,1e-3,1e-1,1e-1],wavelength[401:3601,order,0],spect[401:3601,order,0],spect_err[401:3601,order,0],temp_func, return_spec = True)

        plt.figure()
        plt.plot(wavelength[401:3601,order,0],spect[401:3601,order,0], label = 'Original Spectrum')
        plt.plot(wavelength[401:3601,order,0],sp, label = 'Fitted Spectrum')
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Flux')
        plt.title(fit)
        plt.legend(loc='best')
        plt.show()
        
    #FIXME: There should be a better way than this of dealing with "bad" orders.
    orders= list(range(2,15))
    orders.extend(list(range(17,40)))
    order_ind = 0
    orders = [25] #!!! Temp
    for i in orders:
        #FIXME: The template can not have NaNs in it! Just make it smooth over gaps.
        if np.sum(np.isnan(Tau_Ceti_Template[0].data[:,i]) > 0):
            raise UserWarning("Can not have NaNs in template!")
        temp_wave = Tau_Ceti_Template[1].data[:,i]
        temp_spec = Tau_Ceti_Template[0].data[:,i] 
        temp_func = InterpolatedUnivariateSpline(temp_wave, temp_spec, k=1) 
        temp_lwave = np.log(temp_wave)
        temp_dlwave = temp_lwave[1]-temp_lwave[0]
        rvs = []
        rv_errs = []
        for j in range(19):
            spect_mask = np.isnan(spect[501:3350,i,j])
            spect_wave = wavelength[501:3350,i,j][~spect_mask]
            spect_spec = spect[501:3350,i,j][~spect_mask]
            spect_err_ = spect_err[501:3350,i,j][~spect_mask]
            spect_lwave = np.log(spect_wave)
            
            # changing intial conditions changes answer and MSE 
            a = optimise.leastsq(rv_fitting_eqn_old,x0 = [-24,1e-3,1e-3,1e-3], args=(spect_wave, spect_spec, spect_err_, temp_func), epsfcn=1e-6, full_output = True, ftol=1e-6, gtol=1e-6)
            
#rv_fitting_eqn(params, lwave, spect, spect_err, template, lwave0, dlwave, return_spec = False):
            #a = optimise.leastsq(rv_fitting_eqn,x0 = [-22,1e-3,1e-3,1e-3], args=(spect_lwave, spect_spec, spect_err_, temp_spec, temp_lwave[0], temp_dlwave), \
             #   epsfcn=1e-4, full_output = True, ftol=1e-7, gtol=1e-7)      

            print(a[0])
            
            if False:
                plt.figure()
                plt.plot(spect_wave,a[2]['fvec'])
                plt.show()
            #import pdb; pdb.set_trace()

            #!!! To be neatened. All velocities should be saved in a 2D array !!!
            if a[0][0] != -24 and not (a[1] is None):
               rvs.append(a[0][0])
               #Multiply the error but the square root of chi-squared. !!! Why is chi-squared so high? !!!
               rv_errs.append(np.sqrt(np.mean(a[2]['fvec']**2))*np.sqrt(a[1][0,0]))

        rvs = np.array(rvs)
        rv_errs = np.array(rv_errs)
        print(np.median(rvs))
        weights = 1/rv_errs**2
        wtmn_rv = np.sum(weights*rvs)/np.sum(weights)
        wtmn_rv_err = 1/np.sqrt(np.sum(weights))
        print("Weighted mean RV (km/s): {:.4f} +/- {:.4f}".format(wtmn_rv, wtmn_rv_err))
        #The mean square error will be significantly larger than 1 (e.g. more than 1.5) if fiber to fiber 
        #variations are determined by more than just random errors.
        print("MSE: {:.2f}".format(np.mean((wtmn_rv - rvs)**2/rv_errs**2)))
        if False:
            plt.figure()
            plt.errorbar(np.arange(len(rvs))+1, rvs, rv_errs, fmt='.')
            plt.title('Order:' + str(i))
            plt.xlabel('Fiber num')
            plt.ylabel('Velocity (km/s)')
            plt.show()
        
        
        BC_t, BC_star = barycentric_correction('11dec30096o.fits',obs_file_path[48:58]+'o.fits')
        
        print('True Velocity ', BC_star*c.c.to(u.km/u.s).value)
        print('Velocity Difference (m/s) ', wtmn_rv*1000 - BC_star*c.c.to(u.m/u.s).value)

        print('Percentage Error ', 100*(wtmn_rv - BC_star*c.c.to(u.km/u.s).value)/(BC_star*c.c.to(u.km/u.s).value))
        velocity_err[file_ind, order_ind] = (wtmn_rv*1000 - BC_star*c.c.to(u.m/u.s).value)
        order_ind += 1
    file_ind += 1

if False:
    for order in range(36):        
        plt.figure()   
        plt.plot(velocity_err[:,order])
        plt.show()
print('Velocity uncertainty, orders 89 to 99: {:.1f}'.format(np.std(np.mean(velocity_err[:, 24:34], axis=1))))

