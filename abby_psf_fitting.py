from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


from matplotlib import colors
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from astropy.visualization import simple_norm
from photutils.datasets import (load_simulated_hst_star_image,
                                make_noise_image)
#from photutils.detection import find_peaks
from scipy.signal import find_peaks
from photutils.psf import EPSFBuilder, extract_stars, PSFPhotometry

from matplotlib import colormaps
import matplotlib.colors as colors
import pdb
plt.ion()

data_dir = '/Users/mireland/data/veloce/'

#Code for loading fits files

def load_fits_data(path):
    """
    Args: path is a string of the dirctory to the fits file
    returns a numpy array of the image
    """

    return fits.getdata(path, ignore_missing_simple = True)

def load_fits_header(path):
   """
    Args: path is a string of the dirctory to the fits file
    returns a numpy array of the header
    """
   
   with fits.open(path, ignore_missing_simple=True) as hdul:
        header = hdul[0].header
        return header
        
        
#Dodgy extraction of the spectrum

def extraction(x_positions, y_positions):
    """
    Args:
        x_positions a 1D numpy array with the x pixel locations of the track (length 3900)
        y_positions a 1D numpy array with the y pixel locations of the track (length 3900)

    Returns: 
    extraction, a 1D numpy array of length 3900 with the corresponding value in the image at each (x_pixel, y_pixel location)

    """

    assert np.shape(x_positions) == np.shape(y_positions)
    extraction = np.array([])

    for x_position, y_position in zip(x_positions, y_positions):
        extraction = np.append(extraction, [data[y_position][x_position]])

    return extraction

def extraction_with_buffer(x_positions, y_positions, buffer):
    """
    Args:
        x_positions a 1D numpy array with the x pixel locations of the track (length 3900)
        y_positions a 1D numpy array with the y pixel locations of the track (length 3900)
        buffer (int), the number of pixels on either side of the track to include
    
    Returns:
        total_extraction, a 1D numpy array of length 3900, the sum of the extraction for every pixel in the buffer.
        The sum of calling extraction on (x_position +/- buffer, y_position)
    """
    total_extraction = np.zeros((3900,))
    for i in range(-1*buffer, buffer + 1):
        total_extraction += extraction(x_positions + i, y_positions)
    return total_extraction
    
    
#Second peak finding algorithm

def find_peaks_1(spectra, num_peaks=200):
    """
    Args:
        spectra, a 1D numpy array
        num_peaks, int of the number of peaks to find

    Returns:
        sorted_peaks, list of length num_peaks, the indicies in the spectra of the locations of the num_peaks brightest peaks

    Finds peaks by finding locations where the pixel value is greater than the pixel on either side of it
    Estimates the peak location using a quadratic np.polyfit
    """

    peaks = []
    peak_values = []
    n = len(spectra)
    
    #!!! This is a slow for loop through all points in the 1D spectrum
    for i in range(1, n-1):
        if spectra[i] > spectra[i-1] and spectra[i] > spectra[i+1]:
            x_vals = [i-1, i, i+1]
            x0, x1, x2 = x_vals
            y0, y1, y2 = [spectra[x0], spectra[x1], spectra[x2]]
            max_x = x1 - 0.5 + (y1 - y0)/(2*y1-y2-y0)
            peaks.append(max_x)
            max_value = np.interp(max_x, np.arange(np.shape(spectra)[0]), spectra)
            peak_values.append(max_value)
           

    peak_values = np.array(peak_values)
    peaks = np.array(peaks)

    peak_median_value = np.median(peak_values)

    peak_values_to_return = []
    peaks_to_return = []

    #Here we remove the stupidly bright peaks due to bad pixels etc.
    for peak_loc, peak_value in zip(peaks, peak_values):
        if peak_value < 3*peak_median_value:

            peaks_to_return.append(peak_loc)
            peak_values_to_return.append(peak_value)
    
    #Find the brightest peark_values_to_return peaks.
    indicies = np.flip(np.argsort(np.array(peak_values_to_return)))

    return [peaks_to_return[j] for j in indicies[0:min(num_peaks, len(indicies) -1 )]]

def compute_weighted_x_positions(lfc_x_positions, y_positions, extractions, buffer):
    """
    !!! This probably isn't needed, but is the sort of code you might use to double-check
    and verify the x position of tracks !!!
    
    Args:
        lfc_x_positions: a list of the xpixels for each lfc track (x_positions is length 40 and each item is a 1D np.array of length 3900)
        y_positions: a 1D np.array of the ypixels of lengh 3900 (is the same for every track) 
        extractions: a list of the extracted spectra, spectra are 1D numpy arrays of length 3900
        buffer: int, the buffer that was used in the extraction

    Returns: 
        weighted_x_positions: a list of 40 (one for each track) 1D numpy arrays of length 3900. 
        Represents the weighted x_position when the buffer is included. Weighting is based on the flux in each pixel
    """

    weighted_x_positions = []
    for x_positions, extraction in zip(lfc_x_positions, extractions):
    
        assert np.shape(x_positions) == np.shape(extraction)
        assert np.shape(x_positions) == np.shape(y_positions)

        these_weighted_x_positions = np.array([]) #initialize a new empty array of weighted x values for every x_positions array


        for mid_x_position, y_position, total_flux in zip(x_positions, y_positions, extraction):
            
            to_continue = False
            this_weighted_position = 0
            weight_sum = 0
            this_total_flux = 0

            for i in range(-1*buffer, buffer + 1):

                this_flux = int(data[y_position][mid_x_position + i])
                if this_flux == 0:
                    these_weighted_x_positions = np.append(these_weighted_x_positions, [mid_x_position])
                    to_continue == True
                    break


                this_weighted_position += (mid_x_position + i) * (data[y_position][mid_x_position + i])/total_flux
                weight_sum += (data[y_position][mid_x_position + i])/total_flux
                this_total_flux = this_total_flux + this_flux
            
            if to_continue:
                continue
             
            assert weight_sum < 1.0 + 1*10**6
            assert weight_sum > 1.0 - 1*10**6
            assert this_weighted_position >= mid_x_position - buffer
            assert this_weighted_position <= mid_x_position + buffer
            assert float(this_total_flux) == total_flux

            these_weighted_x_positions = np.append(these_weighted_x_positions, [this_weighted_position])

        weighted_x_positions.append(these_weighted_x_positions)
    
    weighted_pos_fig = plt.figure()
    plt.imshow(data, vmax = 1000)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    for i in range(40):
        plt.scatter(weighted_x_positions[i], y_positions, color = "black", s=2)
        plt.plot(track_data[:, i, lfc_fiber_index] - 1, y_positions, color = "orange")
    plt.title("Diagnostic Plot: compute_weighted_x_positions")

    return weighted_x_positions

#------------------------------------------

#Load an example image, and bias subtract it.
data = load_fits_data(data_dir + "31aug30045.fits")

#Background subtract
        #the stars cutouts used for the ePSF must be background subtracted
        #split the image into quadrants and subtract the 25th percentile
#data = data.astype(np.uint16)  # Ensure the example data is in uint16 type

(y_dim, x_dim) = np.shape(data)                            
q00_bkg = np.percentile(data[:y_dim//2, :x_dim//2], 25) 
q01_bkg = np.percentile(data[y_dim//2:, :x_dim//2], 25) 
q10_bkg = np.percentile(data[:y_dim//2, x_dim//2:], 25) 
q11_bkg = np.percentile(data[y_dim//2:, x_dim//2:], 25) 

#data = data.astype(np.float64)
data = data.astype(float)

data[:y_dim//2,:x_dim//2] -= q00_bkg 
data[y_dim//2:,:x_dim//2] -= q01_bkg 
data[:y_dim//2, x_dim//2:] -= q10_bkg 
data[y_dim//2:, x_dim//2:] -= q11_bkg 
data = data.clip(min=-10) #Clip off the really negative pixels

#Load in the track file
track_data = load_fits_data(data_dir + "31aug30045_0053oi_track.fits")

num_orders = track_data.shape[1] #there are 40 orders
num_fibers = track_data.shape[2] #there are 26 fibers
lfc_fiber_index = 25 #fiber 26 (index 25) is the LFC

all_x_positions = [] #a list of the xpixels for each track (x_positions is length 1040 and each item is a 1D np.array of length 3900) 
y_positions = np.arange(3900) + 99 #a 1D np.array of the ypixels of lengh 3900 (is the same for every track)

#Pick out only the LFC traces. Chris Tinney's arrays start with 1 not 0. So we have to subtract 1
weighted_x_positions = track_data[:,:,lfc_fiber_index].T - 1
lfc_x_positions = np.round(weighted_x_positions).astype(int)

lfc_tracks_fig = plt.figure()
plt.clf()
plt.imshow(data, vmax = 1000)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.title('LFC Tracks')
for i in range(num_orders):
	plt.plot(lfc_x_positions[i], y_positions, 'orange')

buffer = 2 #number of pixels on either side of the track to include
extractions = [extraction_with_buffer(x_positions, y_positions, buffer) for x_positions in lfc_x_positions]

extraction_fig = plt.figure()
plt.clf()
plt.plot(extractions[0])
plt.title("First spectra")

#Find the peaks
peaks1 = find_peaks_1(extractions[1])
peak_indices = [find_peaks_1(spectra) for spectra in extractions]

#Don't use this!
#weighted_x_positions = compute_weighted_x_positions(lfc_x_positions, y_positions, extractions, buffer)

#Find the peaks
x_peak_locs = []
y_peak_locs = []

for i in range(40):
    these_x_peak_locs = []
    these_y_peak_locs = []
    #these_x_positions = lfc_x_positions[i]
    these_weighted_x_positions = weighted_x_positions[i]
    #print("these_x_positions", these_x_positions)
    #print("these_weighted_x_positions", these_weighted_x_positions)
    #print("peak_indices", peak_indices[0])
    #interpolate to find value of weighted_x_positions at the peak_index index (float)

    for peak_index in peak_indices[i]:
        #print("peak_index", peak_index)
        index = int(np.floor(peak_index))
        alpha = peak_index - index
        interpolated_x_value = (1-alpha)*these_weighted_x_positions[index] + alpha*these_weighted_x_positions[index + 1]
        interpolated_y_value = (1-alpha)*y_positions[index] + alpha*y_positions[index + 1]
        these_x_peak_locs.append(interpolated_x_value)
        these_y_peak_locs.append(interpolated_y_value)

    x_peak_locs.append(these_x_peak_locs)
    y_peak_locs.append(these_y_peak_locs)
    
#Do real fitting
#Fit the whole image with the same model, using photutils!
#Inputs are:
#1) An astropy Table() containing the x and y values of the peaks.
#2) A size to extract the size.

all_x_peak_locs = np.array([])
all_y_peak_locs = np.array([])


for x_peaks, y_peaks in zip(x_peak_locs, y_peak_locs):
    all_x_peak_locs = np.append(all_x_peak_locs, x_peaks)
    all_y_peak_locs = np.append(all_y_peak_locs, y_peaks)

sources_fig = plt.figure()
plt.imshow(data, vmax = 300, alpha = 0.75)
plt.scatter(all_x_peak_locs, all_y_peak_locs, color = "pink", s=20, linewidths = 1, alpha = 1)
plt.show(sources_fig)




stars_tbl = Table()
stars_tbl["x"] = all_x_peak_locs
stars_tbl["y"] = all_y_peak_locs


#Create cutouts using extract_stars()

nddata = NDData(data=data) #extract stars requires input data as a NDData object
stars = extract_stars(nddata, stars_tbl, size=7) #stars are a EPSFStar object

#Construct the ePSF with the EPSFBuilder class

epsf_builder = EPSFBuilder(oversampling=2, maxiters=10,
                           progress_bar=False) #should generally use at least 10 iters

#This is the line that does everything (which is why there is a progress bar!)
epsf, fitted_stars = epsf_builder(stars)

norm = simple_norm(epsf.data, 'log', percent=99.0)
all_epsf_fig = plt.figure()
plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.show(all_epsf_fig)


init_params = Table()
init_params["x_0"] = all_x_peak_locs
init_params["y_0"] = all_y_peak_locs
psf_photometry = PSFPhotometry(psf_model = epsf, fit_shape = (5,5), aperture_radius = 1, finder = None)

results = psf_photometry(data, init_params=init_params)
print(results.keys())
print(results)

resid_image = psf_photometry.make_residual_image(data, psf_shape = (5,5))

all_resid_fig = plt.figure()
plt.imshow(resid_image, norm=colors.LogNorm())

