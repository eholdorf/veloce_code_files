import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time

dd = Table.read('data/taucet.csv')
times = Time(dd['MJD'], format='mjd').to_datetime()
plt.clf()
dd['rv'] -= np.mean(dd['rv'])
plt.errorbar(times, dd['rv'], yerr=dd['err'], fmt='.', capsize=5)
plt.ylabel('RV (m/s)')
plt.xlabel('Date')
plt.tight_layout()
plt.show()
