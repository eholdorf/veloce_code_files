import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import astropy.io.fits as pyfits

# Tau Ceti (HD10700) - closest sun like star
TC_16nov18_30126 = pyfits.open('/priv/avatar/velocedata/Data/spec_211202/181116/16nov30126oi_extf.fits')

TC_16nov18_30126_header = pyfits.getheader('/priv/avatar/velocedata/Data/spec_211202/181116/16nov30126oi_extf.fits')

# plotting spectrum fibre by fibre for all orders
for j in range(6):#range(np.shape(TC_16nov18_30126[1])[2]):
	data_legend = mlines.Line2D([], [], color='k', label='Data')
	error_legend = mlines.Line2D([], [], color='r', label='Error')

	plt.plot(TC_16nov18_30126[1].data[:,:,j],TC_16nov18_30126[0].data[:,:,j],'k',label = 'Data')
	plt.plot(TC_16nov18_30126[1].data[:,:,j],TC_16nov18_30126[2].data[:,:,j],'r',label = 'Error')
	plt.title("Tau Ceti (16nov18, 30126) - Fibre "+str(j+1))
	plt.xlabel("$\lambda$ ($\AA$)")
	plt.ylabel("Flux")
	plt.legend(handles = [data_legend, error_legend],loc = 'upper right')
	plt.show()


# HD85512 - very stable M0 star

HD85512_02oct19_30169 = pyfits.open('/priv/avatar/velocedata/Data/spec_211202/191002/02oct30169oi_extf.fits')

HD85512_02oct19_30169_header = pyfits.getheader('/priv/avatar/velocedata/Data/spec_211202/191002/02oct30169oi_extf.fits')

# plotting spectrum fibre by fibre for all orders
for j in range(6):#range(np.shape(HD85512_02oct19_30169[1])[2]):
	data_legend = mlines.Line2D([], [], color='k', label='Data')
	error_legend = mlines.Line2D([], [], color='r', label='Error')

	plt.plot(HD85512_02oct19_30169[1].data[:,:,j],HD85512_02oct19_30169[0].data[:,:,j],'k',label = 'Data')
	plt.plot(HD85512_02oct19_30169[1].data[:,:,j],HD85512_02oct19_30169[2].data[:,:,j],'r',label = 'Error')
	plt.title("HD85512 (02oct19, 30169) - Fibre "+str(j+1))
	plt.xlabel("$\lambda$ ($\AA$)")
	plt.ylabel("Flux")
	plt.legend(handles = [data_legend, error_legend],loc = 'upper right')
	plt.show()
