# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:00:14 2025

@author: maboorai279
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt


def histogramme(ima):
    h = np.zeros([256, 1])  
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


img_avant_b = cv2.imread("2023-01-29-00_00_2023-01-29-23_59_Sentinel-2_L2A_B02_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_v = cv2.imread("2023-01-29-00_00_2023-01-29-23_59_Sentinel-2_L2A_B03_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_r = cv2.imread("2023-01-29-00_00_2023-01-29-23_59_Sentinel-2_L2A_B04_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_pir = cv2.imread("2023-01-29-00_00_2023-01-29-23_59_Sentinel-2_L2A_B08_(Raw).tiff", cv2.IMREAD_GRAYSCALE)

dim_avant = img_avant_r.shape

image_avant = np.zeros([dim_avant[0], dim_avant[1], 4], dtype = 'uint8')
image_avant[:, :, 0] = img_avant_b
image_avant[:, :, 1] = img_avant_v
image_avant[:, :, 2] = img_avant_r
image_avant[:, :, 3] = img_avant_pir

image_avant_col = image_avant[:, :, 0:3]
cv2.imshow("image en hiver en vraie couleur", image_avant_col*4)
cv2.waitKey(0)

image_avant_col[:, :, 2] = img_avant_pir
cv2.imshow("image en hiver en fausse couleur", image_avant_col*2)
cv2.waitKey(0)

cv2.imshow("image en hiver en gris", image_avant_col[:, :, 2])
cv2.waitKey(0)

h = histogramme(image_avant_col[:, :, 2])
plt.Figure()
plt.plot(h)
plt.title('Histogramme en hiver')
plt.xlabel('Valeurs de pixels')
plt.ylabel('Fréquence')
plt.show()

image_avant_bin = binarisation(image_avant_col[:, :, 2], 5)
cv2.imshow("image en hiver apres binarisation", image_avant_bin*255)
cv2.waitKey(0)

# La fonction renvoie un label pour chaque composante et le nombre de composantes
n_avant, labels_avant = cv2.connectedComponents(image_avant_bin)

# Affichage du nombre de régions (composantes connectées)
print("Nombre de régions connectées en hiver : ", n_avant)


img_apres_b = cv2.imread("2023-08-22-00_00_2023-08-22-23_59_Sentinel-2_L2A_B02_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_v = cv2.imread("2023-08-22-00_00_2023-08-22-23_59_Sentinel-2_L2A_B03_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_r = cv2.imread("2023-08-22-00_00_2023-08-22-23_59_Sentinel-2_L2A_B04_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_pir = cv2.imread("2023-08-22-00_00_2023-08-22-23_59_Sentinel-2_L2A_B08_(Raw).tiff", cv2.IMREAD_GRAYSCALE)

dim_apres = img_apres_r.shape

image_apres = np.zeros([dim_apres[0], dim_apres[1], 4], dtype = 'uint8')
image_apres[:, :, 0] = img_apres_b
image_apres[:, :, 1] = img_apres_v
image_apres[:, :, 2] = img_apres_r
image_apres[:, :, 3] = img_apres_pir

image_apres_col = image_apres[:, :, 0:3]
cv2.imshow("image apres en vraie couleur", image_apres_col*4)
cv2.waitKey(0)

image_apres_col[:, :, 2] = img_apres_pir
cv2.imshow("image en été en fausse couleur", image_apres_col*2)
cv2.waitKey(0)

cv2.imshow("image en été en gris", image_apres_col[:, :, 2])
cv2.waitKey(0)

h = histogramme(image_apres_col[:, :, 2])
plt.Figure()
plt.plot(h)
plt.title('Histogramme en été')
plt.xlabel('Valeurs de pixels')
plt.ylabel('Fréquence')
plt.show()

image_apres_bin = binarisation(image_apres_col[:, :, 2], 5)
cv2.imshow("image en été apres binarisation", image_apres_bin*255)
cv2.waitKey(0)

# La fonction renvoie un label pour chaque composante et le nombre de composantes
n_apres, labels_apres = cv2.connectedComponents(image_apres_bin)

# Affichage du nombre de régions (composantes connectées)
print("Nombre de régions connectées en été : ", n_apres)

cv2.destroyAllWindows()

print("Nombre de de bateaux autour de Port-Cros est : ", n_apres - n_avant)