import numpy as np
import pywt

def haar_wavelet(data):
    col = data.shape[2]
    # print("col", col)
    row = data.shape[1]
    # print("col", row)
    d3= data.shape[0]
    # print("col", d3)
    finaldata= np.empty((d3,row*2,col))
    print(data.shape)
    for j in range(d3):
        new_data= data[j,:,:].copy()
        # print(new_data.shape)
        for i in range(row):
            new_row = new_data[i,:].copy()
            # print(new_row.shape)
            (cA, cD) = pywt.dwt(new_row, 'haar')
            # print(new_col.shape)
            # print(cA.shape)
            # print(cD.shape)
            new_data1 = np.concatenate((cA,cD),0)
            # print(new_data1.shape)
            new_data2 = np.reshape(new_data1,(-1,col))
            print(new_data2.shape)
            new_data = np.vstack((new_data,new_data2))
            print(new_data.shape)
        finaldata[j,:,:]=new_data
    print(finaldata.shape)

    return finaldata

