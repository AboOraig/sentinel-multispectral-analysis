# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:39:08 2025

@author: maboorai279
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d

def histogramme(ima):
    h = np.zeros([65535, 1])  
    NL, NC = ima.shape 
    for i in range(NL):
        for j in range(NC):
            val = ima[i, j]  
            h[val] += 1 
    return h

def binarisation(img_in, seuil):
    dim = img_in.shape
    img_out = np.zeros(dim)
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            if img_in[i,j] > seuil:
                img_out[i,j] = 1
                
    img_out = img_out.astype('uint8')                
    return img_out

def trans_lin(ima):
    ima = ima.astype('float')
    
    xmin = ima.min()
    xmax = ima.max()
    
    a = 255 / (xmax - xmin)
    b = -(255 * xmin) / (xmax - xmin)
    
    ima_trans = (a * ima) + b
    ima_trans = ima_trans.astype('uint8')
    
    return ima_trans
    

img_3 = cv2.imread("2018-08-09-00_00_2018-08-09-23_59_Sentinel-3_OLCI_B03_(Raw).tiff", cv2.IMREAD_UNCHANGED)
img_6 = cv2.imread("2018-08-09-00_00_2018-08-09-23_59_Sentinel-3_OLCI_B06_(Raw).tiff", cv2.IMREAD_UNCHANGED)
img_10 = cv2.imread("2018-08-09-00_00_2018-08-09-23_59_Sentinel-3_OLCI_B10_(Raw).tiff", cv2.IMREAD_UNCHANGED)
img_11 = cv2.imread("2018-08-09-00_00_2018-08-09-23_59_Sentinel-3_OLCI_B11_(Raw).tiff", cv2.IMREAD_UNCHANGED)
img_12 = cv2.imread("2018-08-09-00_00_2018-08-09-23_59_Sentinel-3_OLCI_B12_(Raw).tiff", cv2.IMREAD_UNCHANGED)

dim = img_3.shape

image = np.zeros([dim[0], dim[1], 5], dtype = 'uint16')
image[:, :, 0] = img_3
image[:, :, 1] = img_6
image[:, :, 2] = img_10
image[:, :, 3] = img_11
image[:, :, 4] = img_12

image_col = np.zeros([dim[0], dim[1], 3], dtype = 'uint8')
image_col[:, :, 0] = trans_lin(image[:, :, 0])
image_col[:, :, 1] = trans_lin(image[:, :, 1])
image_col[:, :, 2] = trans_lin(image[:, :, 2])

cv2.imshow("image en vraie couleur", image_col*2)
cv2.waitKey(0)

image_col[:, :, 2] = trans_lin(image[:, :, 4])
cv2.imshow("image en fausse couleur", image_col)
cv2.waitKey(0)

h = histogramme(image[:, :, 4])
plt.plot(h)
plt.axis([0, 10000, 1, 1500]) 

masque_terre_nuage = binarisation(image[:, :, 4], 4000)
cv2.imshow("masque terre-nuage", masque_terre_nuage*255)
cv2.waitKey(0)

kernel = np.ones((5,5), dtype=np.uint8)
image_dilate = cv2.dilate(masque_terre_nuage, kernel)

cv2.imshow("image apres dilatation", image_dilate*255)
cv2.waitKey(0)

masque_eau = 1 - image_dilate

cv2.imshow("masque eau", masque_eau*255)
cv2.waitKey(0)

image = image.astype('float')
MCI = image[:, :, 3] - (image[:, :, 2] + (image[:, :, 4] - image[:, :, 2]) * (709 - 681) / (754 - 681))
MCI = MCI * masque_eau
MCI_colored = cv2.applyColorMap(trans_lin(MCI), cv2.COLORMAP_JET)

cv2.imshow("MCI avec colormap", MCI_colored)
cv2.waitKey(0)

MCI_bg = medfilt2d(MCI, kernel_size = 51)
MCI_bg_colored = cv2.applyColorMap(trans_lin(MCI_bg), cv2.COLORMAP_JET)

cv2.imshow("MCI_bg avec colormap", MCI_bg_colored)
cv2.waitKey(0)

dMCI = MCI - MCI_bg
dMCI_colored = cv2.applyColorMap(trans_lin(dMCI), cv2.COLORMAP_JET)

masque_terre_nuage_colored = cv2.merge([masque_terre_nuage * 255] * 3)
cv2.imshow("dMCI avec colormap", dMCI_colored + masque_terre_nuage_colored)
#cv2.imshow("dMCI avec colormap", dMCI_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()