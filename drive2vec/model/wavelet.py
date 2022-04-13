import numpy as np
import pywt

def haar_wavelet(data):
    col = data.shape[1]
    row = data.shape[0]
    for i in range(col):
        new_col = data[:,i].copy()
        (cA, cD) = pywt.dwt(new_col, 'haar')
        new_data = np.concatenate((cA,cD), 0),(row,1)
        new_data = np.reshape(new_col)
        new_data = np.hstack((data,new_col))
    return new_data