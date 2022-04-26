"""
Some utility functions to help with the RV reduction
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
from astropy.table import Table
import os

def correct_bad(in_spect, bad_mask, Bplus=None, max_ftpix = 64):
    """
    Correct a spectrum using the `Mike Ireland bad pixel`
    matrix pseudo-inverse method.
    
    Parameters
    ----------
    in_spect: numpy array
        Input spectrum
    bad_mask: numpy array
        A mask that is 0 for good elements and 1 for elements you want to
        paste over
    Bplus: numpy array
        An internal variable that takes some time to compute. If doing many 
        computations with the same bad_mask, you can comput this only once.
    max_ftpix: int
        The maximum spectral frequency in cycles per 2x the spectrum length
        that should be present in the data.
    """
    #First, extend the bad_mask and in_spect.
    good_pix = np.where(bad_mask==0)[0]
    start_line = np.polyfit(good_pix[:8], in_spect[good_pix[:8]],1)
    end_line = np.polyfit(len(in_spect)-good_pix[-8:], in_spect[good_pix[-8:]],1)
    start = start_line[-1]
    end = end_line[-1]
    offset = start + (end-start)*np.arange(len(in_spect))/len(in_spect)
    ext_spect = np.concatenate((in_spect-offset, (offset-in_spect)[::-1]))
    ext_bad = np.concatenate((bad_mask, bad_mask[::-1]))
    
    #Phew. Now fcreate the big bad matrix
    zeroft = np.arange(max_ftpix, len(ext_spect)//2)
    bad = np.where(ext_bad)[0]
    
    if Bplus is None:
        nbad = len(bad)
        Bz = np.zeros((nbad,len(zeroft)*2))
        # Create a 1D grid. 
        x_ix = 2*np.pi*np.arange(len(ext_spect)//2 + 1)/float(len(ext_spect))
        for i in range(nbad):
            bft = np.exp(-1j*(bad[i]*x_ix))
            Bz[i] = np.append(bft[zeroft].real, bft[zeroft].imag)
        #See AOInstrument
        hb = np.transpose(np.conj(Bz))
        Bplus = np.dot(hb,np.linalg.inv(np.dot(Bz,hb)))
    
    #Now find the image Fourier transform on the "zero" region in the Fourier plane
    #To minimise numerical errors, set the bad pixels to zero at the start.
    ext_spect[bad]=0    
    ft_spect = (np.fft.rfft(ext_spect))[zeroft]

    # Now compute the bad pixel corrections. (NB a sanity check here is
    # that the imaginary part really is 0)
    addit = -np.real(np.dot(np.append(ft_spect.real, ft_spect.imag),Bplus))

    ext_spect[bad] += addit

    return ext_spect[:len(in_spect)] + offset, Bplus
     
#Create local global variables in this script.
thisdir = os.path.dirname(os.path.abspath(__file__))
allzero = Table.read(thisdir + '/../data/vccf_allzero.txt', format='ascii')
ifuzero = Table.read(thisdir + '/../data/vccf_ifuzero.txt', format='ascii')
orderslope = Table.read(thisdir + '/../data/vccf_orderslope.txt', format='ascii')
lcoffset = Table.read(thisdir + '/../data/vccf_lcoffset.txt', format='ascii')
lcoffset_lis = []
#Convert this to a list of lists.
MIN_ORDER = 67
for i in range(MIN_ORDER,103):
    ww = np.where(lcoffset['m']==i)[0]
    lcoffset_lis += [[lcoffset['MJD'][ww].data, lcoffset['offset'][ww].data]]

    
def correct_lc_rv_chris(mjd, min_order=67, max_order=102, return_all=False):
    """Based on Chris Tinney's computations, compute the corrections as a
    function of order"""
    if min_order < MIN_ORDER:
        raise UserWarning("Invalid minimum order for Veloce Rosso!")
        
    norders = max_order - min_order + 1

    slope_offsets = np.zeros(norders)
    lc_offsets = np.zeros(norders)

    this_slope = np.interp(mjd, orderslope['MJD'], orderslope['orderslope'])
    zero_offset = np.interp(mjd, allzero['MJD'], allzero['zeropoint'])
    ifu_offset = np.interp(mjd, ifuzero['MJD'], ifuzero['zeropoint'])
    for i in range(norders):
        xy = lcoffset_lis[min_order-MIN_ORDER + i]
        if len(xy[0]) != 0:
            lc_offsets[i] = np.interp(mjd, xy[0], xy[1])
        slope_offsets[i] = (i+min_order-84) * this_slope
        
    if return_all:
        return slope_offsets, lc_offsets, zero_offset, ifu_offset
    else:
        return slope_offsets+ lc_offsets+ zero_offset+ ifu_offset
     

#Here is some code to standalone test this 
if __name__=="__main__":
    x = np.arange(2048)-1299
    y = np.exp(-x**2/500**2)
    bad = np.zeros(2048, dtype=np.int8)
    bad[[5,60,100,130,400,500,501,502,503,1300,1301,1302,1303,1304]]=1
    yorig = y.copy()
    y[bad==1] = 1.1
    corrected, Bplus = correct_bad(y, bad)
    plt.clf()
    plt.plot(y)
    plt.plot(yorig)
    plt.plot(corrected)
    plt.show()
