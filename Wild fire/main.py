# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:33:35 2025

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
            if img_in[i, j] > seuil:  # Seuil défini pour détecter les zones brûlées
                img_out[i, j] = 1  # Mise à 1

    img_out = img_out.astype('uint8')  # Conversion en type entier (uint8)
    return img_out

# Chargement des images Sentinel-2 avant l’incendie (19/06/2017)
# Les images sont chargées en niveaux de gris (grayscale)
img_avant_b = cv2.imread("2017-06-19-00_00_2017-06-19-23_59_Sentinel-2_L2A_B02_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_v = cv2.imread("2017-06-19-00_00_2017-06-19-23_59_Sentinel-2_L2A_B03_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_r = cv2.imread("2017-06-19-00_00_2017-06-19-23_59_Sentinel-2_L2A_B04_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_pir = cv2.imread("2017-06-19-00_00_2017-06-19-23_59_Sentinel-2_L2A_B08_(Raw).tiff", cv2.IMREAD_GRAYSCALE)

# Création d'une image avec 4 bandes spectrales
dim_avant = img_avant_r.shape
image_avant = np.zeros([dim_avant[0], dim_avant[1], 4], dtype = 'uint8')
image_avant[:, :, 0] = img_avant_b
image_avant[:, :, 1] = img_avant_v
image_avant[:, :, 2] = img_avant_r
image_avant[:, :, 3] = img_avant_pir

# Affichage de l'image en vraie couleur (RGB)
image_avant_col = image_avant[:, :, 0:3]
cv2.imshow("image avant l’incendie en vraie couleur", image_avant_col*2)
cv2.waitKey(0)

# Affichage de l'image en fausse couleur (PIR G B)
image_avant_col[:, :, 2] = img_avant_pir
cv2.imshow("image avant l’incendie en fausse couleur", image_avant_col)
cv2.waitKey(0)

# Affichage de la bande proche infrarouge en niveaux de gris
cv2.imshow("image avant l’incendie en gris", image_avant_col[:, :, 2])
cv2.waitKey(0)

# Calcule de la NDVI pour detecter les zones brûlées
img_avant_pir = img_avant_pir.astype(np.float32)
img_avant_r = img_avant_r.astype(np.float32)
ndvi_avant = (img_avant_pir - img_avant_r) / (img_avant_pir + img_avant_r)

# Normalisation du NDVI (de -1 à 1 -> de 0 à 255)
alpha = 255 / (np.max(ndvi_avant) - np.min(ndvi_avant))
beta = -255 * np.min(ndvi_avant) / (np.max(ndvi_avant) - np.min(ndvi_avant))
ndvi_avant_norm = alpha * ndvi_avant + beta
ndvi_avant_norm = ndvi_avant_norm.astype(np.uint8)

# Application du colormap
ndvi_avant_colormap = cv2.applyColorMap(ndvi_avant_norm, cv2.COLORMAP_JET)

# Affichage du NDVI avec colorbar
cv2.imshow("NDVI avant l’incendie avec colormap", ndvi_avant_colormap)
cv2.waitKey(0)

# Calcul de l'histogramme du NDVI normalisé
h = histogramme(ndvi_avant_norm)

# Affichage de l'histogramme
plt.Figure()
plt.plot(h)
plt.title('Histogramme du NDVI avant l’incendie')
plt.xlabel('Valeurs de pixels')
plt.ylabel('Fréquence')
plt.show()

# Application de la binarisation sur l'image avant l’incendie (seuil = 150)
ndvi_avant_bin = binarisation(ndvi_avant_norm, 160)
cv2.imshow("NDVI binaire avant l'incendie", ndvi_avant_bin * 255)
cv2.waitKey(0)


# Chargement des images Sentinel-2 apres l’incendie (29/07/2017)
# Les images sont chargées en niveaux de gris (grayscale)
img_apres_b = cv2.imread("2017-07-29-00_00_2017-07-29-23_59_Sentinel-2_L2A_B02_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_v = cv2.imread("2017-07-29-00_00_2017-07-29-23_59_Sentinel-2_L2A_B03_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_r = cv2.imread("2017-07-29-00_00_2017-07-29-23_59_Sentinel-2_L2A_B04_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_apres_pir = cv2.imread("2017-07-29-00_00_2017-07-29-23_59_Sentinel-2_L2A_B08_(Raw).tiff", cv2.IMREAD_GRAYSCALE)

# Création d'une image avec 4 bandes spectrales
dim_apres = img_apres_r.shape
image_apres = np.zeros([dim_apres[0], dim_apres[1], 4], dtype = 'uint8')
image_apres[:, :, 0] = img_apres_b
image_apres[:, :, 1] = img_apres_v
image_apres[:, :, 2] = img_apres_r
image_apres[:, :, 3] = img_apres_pir

# Affichage de l'image en vraie couleur (RGB)
image_apres_col = image_apres[:, :, 0:3]
cv2.imshow("image apres l’incendie en vraie couleur", image_apres_col*2)
cv2.waitKey(0)

# Affichage de l'image en fausse couleur (PIR G B)
image_apres_col[:, :, 2] = img_apres_pir
cv2.imshow("image apres l’incendie en fausse couleur", image_apres_col)
cv2.waitKey(0)

# Affichage de la bande proche infrarouge en niveaux de gris
cv2.imshow("image apres l’incendie en gris", image_apres_col[:, :, 2])
cv2.waitKey(0)

# Calcule de la NDVI pour detecter les zones brûlées
img_apres_pir = img_apres_pir.astype(np.float32)
img_apres_r = img_apres_r.astype(np.float32)
ndvi_apres = (img_apres_pir - img_apres_r) / (img_apres_pir + img_apres_r)

# Normalisation du NDVI (de -1 à 1 -> de 0 à 255)
alpha = 255 / (np.max(ndvi_apres) - np.min(ndvi_apres))
beta = -255 * np.min(ndvi_apres) / (np.max(ndvi_apres) - np.min(ndvi_apres))
ndvi_apres_norm = alpha * ndvi_apres + beta
ndvi_apres_norm = ndvi_apres_norm.astype(np.uint8)

# Application du colormap
ndvi_apres_colormap = cv2.applyColorMap(ndvi_apres_norm, cv2.COLORMAP_JET)

# Affichage du NDVI avec colorbar
cv2.imshow("NDVI apres l’incendie avec colormap", ndvi_apres_colormap)
cv2.waitKey(0)

# Calcul de l'histogramme du NDVI normalisé
h = histogramme(ndvi_apres_norm)

# Affichage de l'histogramme
plt.Figure()
plt.plot(h)
plt.title('Histogramme du NDVI apres l’incendie')
plt.xlabel('Valeurs de pixels')
plt.ylabel('Fréquence')
plt.show()

# Application de la binarisation sur l'image apres l’incendie (seuil = 150)
ndvi_apres_bin = binarisation(ndvi_apres_norm, 160)
cv2.imshow("NDVI binaire apres l'incendie", ndvi_apres_bin * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Comparaison entre les images binaires avant et après l’incendie
diff = ndvi_apres_bin - ndvi_avant_bin # Différence entre les images binaires
dim = diff.shape
nbr = 0

# Comptage des pixels où il y a une différence (zones nouvellement brûlée)
for i in range(dim[0]):
    for j in range(dim[1]):
            if diff[i,j] != 0:
                nbr += 1 

# Calcul de la surface brûlée (chaque pixel représente 10x10 m, conversion en hectares)
surface = nbr * 10 * 10 * 0.0001
print(f"Surface brûlée estimée: {surface} hectares")