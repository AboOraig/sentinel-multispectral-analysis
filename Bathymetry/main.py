# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:02:55 2025

@author: maboorai279
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy.stats import linregress


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

img_b = cv2.imread("image_pleiades_b1_bleue.tif", cv2.IMREAD_UNCHANGED)
img_v = cv2.imread("image_pleiades_b2_verte.tif", cv2.IMREAD_UNCHANGED)
img_r = cv2.imread("image_pleiades_b3_rouge.tif", cv2.IMREAD_UNCHANGED)
img_pir = cv2.imread("image_pleiades_b4_PIR.tif", cv2.IMREAD_UNCHANGED)

dim = img_b.shape

image = np.zeros([dim[0], dim[1], 4], dtype = 'uint16')
image[:, :, 0] = img_b
image[:, :, 1] = img_v
image[:, :, 2] = img_r
image[:, :, 3] = img_pir

image_col = np.zeros([dim[0], dim[1], 3], dtype = 'uint8')
image_col[:, :, 0] = trans_lin(image[:, :, 0])
image_col[:, :, 1] = trans_lin(image[:, :, 1])
image_col[:, :, 2] = trans_lin(image[:, :, 2])

cv2.imshow("image en vraie couleur", image_col*2)
cv2.waitKey(0)

image_col[:, :, 2] = trans_lin(image[:, :, 3])
cv2.imshow("image en fausse couleur", image_col[:,:,2]*2)
cv2.waitKey(0)

h = histogramme(image[:, :, 3])
plt.plot(h)
plt.axis([0, 2000, 1, 12000]) 

masque_terre_nuage = binarisation(image[:, :, 3], 500)
cv2.imshow("masque terre-nuage", masque_terre_nuage*255)
cv2.waitKey(0)

masque_eau = 1 - masque_terre_nuage
cv2.imshow("masque eau", masque_eau*255)
cv2.waitKey(0)

n = 1000
bande_bleue = image[:, :, 0].astype('float')
bande_verte = image[:, :, 1].astype('float')

X = np.log(n * bande_bleue) / np.log(n * bande_verte)
X = X * masque_eau

valide_X = X[~np.isnan(X) & (X > 0) & (X != -np.inf)] 
xmin = valide_X.min()
xmax = valide_X.max()
a = 255 / (xmax - xmin)
b = -(255 * xmin) / (xmax - xmin)
X = (a * X) + b
X = X.astype('uint8')

X_colored = cv2.applyColorMap(X, cv2.COLORMAP_JET)
cv2.imshow("X avec colormap",X_colored)
cv2.waitKey(0)

[y,x,w,h] = cv2.selectROI("Selectionnez la zone de reflexion", image_col)
zone_roi = image[x:x+h,y:y+w,:]
cv2.imshow("Zone ROI", zone_roi)
cv2.waitKey(0)

bande_bleue = zone_roi[:, :, 0].flatten()
bande_verte = zone_roi[:, :, 1].flatten()
bande_rouge = zone_roi[:, :, 2].flatten()
bande_PIR = zone_roi[:, :, 3].flatten()

plt.figure(figsize=(10, 6))

plt.scatter(bande_PIR, bande_bleue, alpha=0.5, c='b', label="Bleu")
plt.scatter(bande_PIR, bande_verte, alpha=0.5, c='g', label="Vert")
plt.scatter(bande_PIR, bande_rouge, alpha=0.5, c='r', label="Rouge")

plt.xlabel("PIR")
plt.ylabel("Intensité")
plt.title("Bandes Bleue, Verte et Rouge en fonction de la Bande PIR")
plt.legend()
plt.show()

image = image.astype('float')

res_b = linregress(bande_PIR, bande_bleue)
a = res_b[0]
b = res_b[1]
y_b = a * bande_PIR + b
bande_bleue_cg = image[:, :, 0] - a * (image[:, :, 3] - bande_PIR.min())

res_v = linregress(bande_PIR, bande_verte)
a = res_v[0]
b = res_v[1]
y_v = a * bande_PIR + b
bande_verte_cg = image[:, :, 1] - a * (image[:, :, 3] - bande_PIR.min())

res_r = linregress(bande_PIR, bande_rouge)
a = res_r[0]
b = res_r[1]
y_r = a * bande_PIR + b
bande_rouge_cg = image[:, :, 2] - a * (image[:, :, 3] - bande_PIR.min())

plt.figure(figsize=(10, 6))

plt.plot(bande_PIR, y_b, c='b', linewidth=2, label="Régression Bleu")
plt.plot(bande_PIR, y_v, c='g', linewidth=2, label="Régression Verte")
plt.plot(bande_PIR, y_r, c='r', linewidth=2, label="Régression Rouge")

plt.xlabel("PIR")
plt.ylabel("Intensité")
plt.title("Régression Linéaire des bandes en fonction de la bande PIR")
plt.legend()
plt.show()

image_apres = np.stack((bande_bleue_cg, bande_verte_cg, bande_rouge_cg), axis=-1)
image_corrigee_display = np.clip(image_apres, 0, 255).astype(np.uint8)

cv2.imshow("Image apres correction", image_corrigee_display)
cv2.waitKey(0)

X_cg = np.log(n * bande_bleue_cg.astype('float')) / np.log(n * bande_verte_cg.astype('float'))
X_cg = X_cg * masque_eau

valide_X_cg = X_cg[~np.isnan(X_cg) & (X_cg > 0) & (X_cg != -np.inf)] 
xmin = valide_X_cg.min()
print("xmin: ", xmin)
xmax = np.mean(X_cg[x:x+h,y:y+w])
print("xmax: ", xmax)
a_cg = 255 / (xmax - xmin)
b_cg = -(255 * xmin) / (xmax - xmin)
X_cg = (a_cg * X_cg) + b_cg
X_cg = X_cg * masque_eau

X_colored = cv2.applyColorMap(X_cg.astype('uint8'), cv2.COLORMAP_JET)
cv2.imshow("X avec colormap apres correction", X_colored)
cv2.waitKey(0)

X_min = valide_X_cg.min()
X_max = np.mean(X_cg[x:x+h,y:y+w])

az = 10 / (X_max - X_min)
bz = -az * X_min
Z = az * X_cg + bz
Z = Z.astype('uint8')
Z = Z * masque_eau

plt.figure(figsize=(8, 6))
img_display = plt.imshow(Z, cmap="jet", vmin=0, vmax=10)
plt.colorbar(label="Profondeur (m)")
plt.title("Carte de profondeur (0m à 10m)")
plt.show()

cv2.destroyAllWindows()