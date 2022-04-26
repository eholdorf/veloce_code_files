"""
Show examples of the Tinney VCCF output.

From veloce.datacentral.org.au :

"Ordres runs from m=65 (red end, order=0) to m=104 (blue end, order index 39)"
Corrections run from m=67 to m=102, hence the offset of 2 below.

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import rv
from astropy.time import Time

taucet = Table.read('data/taucet.csv')

allrv = []
lc = []
slope = []
zero = []
ifu = []
dom = [] #Day of month
for i in range(len(taucet)):
    slope_offsets, lc_offsets, zero_offset, ifu_offset = rv.utils.correct_lc_rv_chris(taucet['MJD'][i], return_all=True)
    zero += [zero_offset]
    ifu += [ifu_offset]  
    lc += [lc_offsets] 
    slope += [slope_offsets]
    allrv += [slope_offsets + lc_offsets + ifu_offset + zero_offset]
    dom += [Time(taucet['MJD'][i],format='mjd').datetime.day]
    
allrv = np.array(allrv)

orders = np.array([6,7,13,14,17,25,26,27,28,30,31,33,34,35,36,37])
plt.figure(1)
plt.clf()
plt.title('Minus Tinney Correction Sum')
plt.imshow(-allrv[:,orders-2])
plt.xticks(np.arange(len(orders)), orders)
words = [str(i)+ 'dec19' for i in dom]
plt.yticks(np.arange(len(words)),words)
plt.colorbar(label='RV Error (m/s)')
plt.tight_layout()