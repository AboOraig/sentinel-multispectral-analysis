# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:08:22 2025

@author: maboorai279
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt

# Fonction pour calculer l'histogramme d'une image en niveaux de gris
def histogramme(ima):
    h = np.zeros([256, 1])  # Initialisation de l'histogramme avec 256 niveaux de gris
    NL, NC = ima.shape  # Récupération des dimensions de l'image
    for i in range(NL):
        for j in range(NC):
            val = ima[i, j]  
            h[val] += 1  # Incrémentation de l'histogramme
    return h

# Fonction de binarisation d'une image en utilisant un seuil donné
def binarisation(img_in, seuil):
    dim = img_in.shape
    img_out = np.zeros(dim)  # Initialisation de l'image binaire à 0

    for i in range(dim[0]):
        for j in range(dim[1]):
            if img_in[i, j] < seuil:  # Seuil défini pour détecter les zones inondées
                img_out[i, j] = 1  # Mise à 1 des pixels détectés comme eau

    img_out = img_out.astype('uint8')  # Conversion en type entier (uint8)
    return img_out

# Chargement des images Sentinel-2 avant l'inondation (16/11/2019)
# Les images sont chargées en niveaux de gris (grayscale)
img_avant_b = cv2.imread("2019-11-16-00_00_2019-11-16-23_59_Sentinel-2_L2A_B02_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_v = cv2.imread("2019-11-16-00_00_2019-11-16-23_59_Sentinel-2_L2A_B03_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_r = cv2.imread("2019-11-16-00_00_2019-11-16-23_59_Sentinel-2_L2A_B04_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_pir = cv2.imread("2019-11-16-00_00_2019-11-16-23_59_Sentinel-2_L2A_B08_(Raw).tiff", cv2.IMREAD_GRAYSCALE)

# Création d'une image avec 4 bandes spectrales
dim_avant = img_avant_r.shape
image_avant = np.zeros([dim_avant[0], dim_avant[1], 4], dtype='uint8')
image_avant[:, :, 0] = img_avant_b  # Bande bleue
image_avant[:, :, 1] = img_avant_v  # Bande verte
image_avant[:, :, 2] = img_avant_r  # Bande rouge
image_avant[:, :, 3] = img_avant_pir  # Bande proche infrarouge (PIR)

# Affichage de l'image en vraie couleur (RGB)
image_avant_col = image_avant[:, :, 0:3]
cv2.imshow("Image avant l'inondation en vraie couleur", image_avant_col*3)
cv2.waitKey(0)

# Affichage de l'image en fausse couleur (PIR G B)
image_avant_col[:, :, 2] = img_avant_pir
cv2.imshow("Image avant l'inondation en fausse couleur", image_avant_col)
cv2.waitKey(0)

# Affichage de la bande proche infrarouge en niveaux de gris
cv2.imshow("Image avant l'inondation en gris", image_avant_col[:, :, 2])
cv2.waitKey(0)

# Calcul et affichage de l'histogramme de la bande proche infrarouge
h = histogramme(image_avant_col[:, :, 2])
plt.plot(h)
plt.title("Histogramme de la bande proche infrarouge avant l'inondation")
plt.xlabel("Niveau de gris")
plt.ylabel("Fréquence")
plt.show()

# Application de la binarisation pour détecter les zones inondées (seuil = 25)
image_avant_bin = binarisation(image_avant_col[:, :, 2], 22)
cv2.imshow("Image avant l'inondation après binarisation", image_avant_bin * 255)  # Multiplication pour l'affichage
cv2.waitKey(0)

# Chargement des images Sentinel-2 après l'inondation (03/12/2019)
img_apres_b = cv2.imread("2019-12-03-00_00_2019-12-03-23_59_Sentinel-2_L2A_B02_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_v = cv2.imread("2019-12-03-00_00_2019-12-03-23_59_Sentinel-2_L2A_B03_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_r = cv2.imread("2019-12-03-00_00_2019-12-03-23_59_Sentinel-2_L2A_B04_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_pir = cv2.imread("2019-12-03-00_00_2019-12-03-23_59_Sentinel-2_L2A_B08_(Raw).tiff", cv2.IMREAD_GRAYSCALE)

# Création d'une image avec 4 bandes spectrales
dim_apres = img_apres_r.shape
image_apres = np.zeros([dim_apres[0], dim_apres[1], 4], dtype='uint8')
image_apres[:, :, 0] = img_apres_b
image_apres[:, :, 1] = img_apres_v
image_apres[:, :, 2] = img_apres_r
image_apres[:, :, 3] = img_apres_pir

# Affichage des images après l'inondation
image_apres_col = image_apres[:, :, 0:3]
cv2.imshow("Image après l'inondation en vraie couleur", image_apres_col*2)
cv2.waitKey(0)

image_apres_col[:, :, 2] = img_apres_pir
cv2.imshow("Image après l'inondation en fausse couleur", image_apres_col)
cv2.waitKey(0)

cv2.imshow("Image après l'inondation en gris", image_apres_col[:, :, 2])
cv2.waitKey(0)

# Calcul et affichage de l'histogramme de la bande proche infrarouge après l'inondation
h = histogramme(image_apres_col[:, :, 2])
plt.plot(h)
plt.title("Histogramme de la bande proche infrarouge apres l'inondation")
plt.xlabel("Niveau de gris")
plt.ylabel("Fréquence")
plt.show()

# Application de la binarisation sur l'image après l'inondation
image_apres_bin = binarisation(image_apres_col[:, :, 2], 22)
cv2.imshow("Image après l'inondation après binarisation", image_apres_bin * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Comparaison entre les images binaires avant et après inondation
diff = image_apres_bin - image_avant_bin  # Différence entre les images binaires
dim = diff.shape
nbr = 0

# Comptage des pixels où il y a une différence (zones nouvellement inondées)
for i in range(dim[0]):
    for j in range(dim[1]):
        if diff[i, j] != 0:
            nbr += 1 

# Calcul de la surface inondée (chaque pixel représente 10x10 m, conversion en hectares)
surface = nbr * 10 * 10 * 0.0001
print(f"Surface inondée estimée: {surface} hectares")