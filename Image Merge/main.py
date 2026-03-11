# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 11:37:37 2025

@author: maboorai279
"""


import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy.stats import linregress


img_b = cv2.imread("OLI_C_Bleu_60m.tif", cv2.IMREAD_UNCHANGED)
img_v = cv2.imread("OLI_C_Vert_60m.tif", cv2.IMREAD_UNCHANGED)
img_r = cv2.imread("OLI_C_Rouge_60m.tif", cv2.IMREAD_UNCHANGED)
img_pan = cv2.imread("OLI_PANCHRO_30m.tif", cv2.IMREAD_UNCHANGED)

# Dimensions 
dim_pan = img_pan.shape
print('Dimension PAN : ', dim_pan)

dim = img_b.shape
print('Dimension COUL : ', dim)

# Images PAN et COUL
COUL = np.zeros([dim[0], dim[1], 3], dtype = 'uint8')
COUL[:, :, 0] = img_b
COUL[:, :, 1] = img_v
COUL[:, :, 2] = img_r

cv2.imshow("Image PAN en niveaux de gris", img_pan*2)
cv2.waitKey(0)

cv2.imshow("Image COUL en couleur", COUL*2)
cv2.waitKey(0)

print('Rapport de resolution : ',  dim_pan[0] / dim[0])

# Image COUL_se 
COUL_se = cv2.resize(COUL, (dim_pan[1], dim_pan[0]))

cv2.imshow("Image COUL_se en couleur", COUL_se*2)
cv2.waitKey(0)

# Image PAN_m 
PAN_m = (COUL_se[:,:,0].astype(np.float32) + COUL_se[:,:,1].astype(np.float32) + COUL_se[:,:,2].astype(np.float32)) / 3
PAN_m = PAN_m.astype(np.uint8)

cv2.imshow("Image PAN_m (60m)", PAN_m*2)
cv2.waitKey(0)

# Régression linéaire et PAN_p
pan_flat = img_pan.flatten()
pan_m_flat = PAN_m.flatten()

res = linregress(pan_m_flat, pan_flat)
a = res[0]
b = res[1]

print('a : ',  a)
print('b : ',  b)

PAN_p = a * img_pan + b
PAN_p = PAN_p.astype(np.uint8)

cv2.imshow("Image PAN_p (60m)", PAN_p*2)
cv2.waitKey(0)

# Image COUL_f
COUL_f = np.zeros([dim_pan[0], dim_pan[1], 3], dtype = 'uint8')
COUL_f[:,:,0] = COUL_se[:,:,0] / PAN_m * PAN_p
COUL_f[:,:,1] = COUL_se[:,:,1] / PAN_m * PAN_p
COUL_f[:,:,2] = COUL_se[:,:,2] / PAN_m * PAN_p

cv2.imshow("Image COUL_f en couleur par Brovey", COUL_f*2)
cv2.waitKey(0)

# Espace HSV
COUL_se_HSV = cv2.cvtColor(COUL_se, cv2.COLOR_BGR2HSV)

cv2.imshow("Image COUL_se en HSV", COUL_se_HSV)
cv2.waitKey(0)

# Régression linéaire et PAN_p
COUL_se_V_flat = COUL_se_HSV[:,:,2].flatten()

res = linregress(COUL_se_V_flat, pan_flat)
a = res[0]
b = res[1]

print('a de HSV : ',  a)
print('b de HSV : ',  b)

PAN_p = a * img_pan + b
PAN_p = PAN_p.astype(np.uint8)

cv2.imshow("Image PAN_p de HSV (60m)", PAN_p*2)
cv2.waitKey(0)

# Espace RGB
COUL_se_HSV[:,:,2] = PAN_p
COUL_f = cv2.cvtColor(COUL_se_HSV, cv2.COLOR_HSV2BGR)

cv2.imshow("Image COUL_f en couleur par HSV", COUL_f*3)
cv2.waitKey(0)

cv2.destroyAllWindows()