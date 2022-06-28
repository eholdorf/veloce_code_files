import utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimise
import astropy.io.fits as pyfits
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as optimise

# want to model each order as a series of lines which have centres and heights which can be varied
def test(sz=64, subsamp=4):
    dd = pyfits.open('/priv/avatar/velocedata/Data/spec_211202/191211/11dec30136_0138oi_extf.fits')
    
    x = dd[1].data[:,13,10]
    y = dd[0].data[:,13,10]
    
    
    ym = np.isnan(y)
    x = x[~ym]
    y = y[~ym]
    
    line = []
    i = 1
    while i <len(y)-1:
        if i < 32:
            med_val = np.median(y[0:64])
            std_val = np.std(y[0:64])
        elif i > len(y)-32:
            med_val = np.median(y[-65:-1])
            std_val = np.std(y[-65:-1])
        else:
            med_val = np.median(y[(i-32):(i+32)])
            std_val = np.std(y[(i-32):(i+32)])
        
        if y[i-1] < y[i] and y[i]>y[i+1] and y[i]>5*std_val+med_val:
            line.append(i)
        i += 1
    #line = np.where(y>(np.median(y)+500))
    #print(np.median(y))
    
    xs = x[line]
    ys = y[line]
    
    plt.figure()
    plt.plot(x,y)
    plt.plot(xs,ys,'.')
    plt.show()
    
    u = np.arange(sz//2 + 1)/sz * subsamp
    
    for i in line:
        print(i)
        if i > 10 and i < len(x)-10:
            x1 = x[(i-10):(i+10)]
            y1 = y[(i-10):(i+10)]
        elif i <=10:
            x1 = x[0:10]
            y1 = y[0:10]
        else:
            x1 = x[-11:-1]
            y1 = y[-11:-1]
        #plt.figure()
        #plt.plot(x1,y1)
        
        f = InterpolatedUnivariateSpline(x1,y1,k=5)
        x1 = np.linspace(min(x1), max(x1),64)
        y1 = f(x1)
        #plt.plot(x1,y1)
        #plt.show()
        
        a = optimise.least_squares(func,x0 = ([1,1,1,1,1,1]),args = (u,x1,y1))
        
        plt.figure()
        plt.plot(y1)
        plt.plot(utils.voigt_like_profile(a.x,u))
        plt.show()
        
def func(params,u,x,y):
    return utils.voigt_like_profile(params,u) - y
    
    
def line_fitting(sz=64, subsamp=4):

    dd = pyfits.open('/priv/avatar/velocedata/Data/spec_211202/191211/11dec30136_0138oi_extf.fits')
    
    x = dd[1].data[847:911,13,10]
    y = dd[0].data[847:911,13,10]
    y /= np.sum(y)
    
    u = np.arange(sz//2 + 1)/sz * subsamp
    
    a = optimise.least_squares(func,x0 = ([0,0,0,0]),args = (u,x,y))
    
    plt.figure()
    plt.plot(y)
    plt.plot(utils.voigt_like_profile(a.x,u))
    plt.show()
     
     

