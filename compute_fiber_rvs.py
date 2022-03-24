#calc_rv_corr(file_path, TC_observation_dir, '/priv/avatar/velocedata/Data/spec_211202/191211/11dec30096oi_extf.fits')

dd = pyfits.open('Tau_Ceti_Template_dec2019_tellcor_1.fits')
temp = dd[0].data[:,13]
wave = dd[1].data[:,13]
all_log_w, all_t_logflux, all_t_logerrflux, all_s_logflux, all_s_logerrflux, [airmass_template, airmass_star] = log_scale_interpolation(testing_temp_files[0],testing_temp_files[1])

data = (np.sum(all_t_logflux[:,13,:],1)/np.median(all_t_logflux[:,13,:]))

diff = temp-data

plt.figure()
plt.plot(wave,data)

plt.figure()
plt.plot(wave,temp)

plt.figure()
plt.plot(wave,diff)

plt.show()

first_line_temp = wave[list(temp).index(min(temp[1000:2000]))]
print(first_line_temp)
first_line_data = wave[list(data).index(min(data[1000:2000]))]
print(first_line_data)

diff = first_line_data - first_line_temp 

v = diff/first_line_temp

print(v)
