# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:21:44 2025

@author: maboorai279
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt


img_avant_b = cv2.imread("Nouveau dossier/2025-01-13-00_00_2025-01-13-23_59_Sentinel-2_L2A_B02_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_v = cv2.imread("Nouveau dossier/2025-01-13-00_00_2025-01-13-23_59_Sentinel-2_L2A_B03_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_r = cv2.imread("Nouveau dossier/2025-01-13-00_00_2025-01-13-23_59_Sentinel-2_L2A_B04_(Raw).tiff", cv2.IMREAD_GRAYSCALE)
img_avant_pir = cv2.imread("Nouveau dossier/2025-01-13-00_00_2025-01-13-23_59_Sentinel-2_L2A_B08_(Raw).tiff", cv2.IMREAD_GRAYSCALE)

dim_avant = img_avant_r.shape

image_avant = np.zeros([dim_avant[0], dim_avant[1], 4], dtype = 'uint8')
image_avant[:, :, 0] = img_avant_b
image_avant[:, :, 1] = img_avant_v
image_avant[:, :, 2] = img_avant_r
image_avant[:, :, 3] = img_avant_pir

image_avant_col = image_avant[:, :, 0:3]
cv2.imshow("image en vraie couleur", image_avant_col*2)
cv2.waitKey(0)

image_avant_col[:, :, 2] = img_avant_pir
cv2.imshow("image en fausse couleur", image_avant_col*2)
cv2.waitKey(0)

# on a au moins 3 classes d’items à cartographier : le sol, la vegetation, l'eau
# on peut egalement destangier 2 types d'eau : eau clair et eau turbide.


# On va sélectionner une ROI pour chaque type de classe (eau, végétation, sol)
sol = cv2.selectROI("Selectionnez la zone de Sol", image_avant)
vegetation = cv2.selectROI("Selectionnez la zone de Vegetation", image_avant)
eau = cv2.selectROI("Selectionnez la zone d'Eau", image_avant)

# Extraction des ROI à partir de l'image (le format ROI est [x, y, w, h])
sol_roi = image_avant[sol[1]:sol[1]+sol[3], sol[0]:sol[0]+sol[2]]
vegetation_roi = image_avant[vegetation[1]:vegetation[1]+vegetation[3], vegetation[0]:vegetation[0]+vegetation[2]]
eau_roi = image_avant[eau[1]:eau[1]+eau[3], eau[0]:eau[0]+eau[2]]

# Affichage des ROI sélectionnées
cv2.imshow("Sol ROI", sol_roi)
cv2.waitKey(0)
cv2.imshow("Vegetation ROI", vegetation_roi)
cv2.waitKey(0)
cv2.imshow("Eau ROI", eau_roi)
cv2.waitKey(0)

# Sauvegarde des ROI sous format .npy
np.save("sol_roi.npy", sol_roi)
np.save("vegetation_roi.npy", vegetation_roi)
np.save("eau_roi.npy", eau_roi)

print("ROIs sauvegardées avec succès !")


# Chargement des ROIs
sol_roi = np.load("sol_roi.npy")
vegetation_roi = np.load("vegetation_roi.npy")
eau_roi = np.load("eau_roi.npy")

# Calcul la moyenne des valeurs de chaque classe en ligne, colonne et pas en bande
sol_mean = np.mean(sol_roi, axis=(0, 1))
vegetation_mean = np.mean(vegetation_roi, axis=(0, 1))
eau_mean = np.mean(eau_roi, axis=(0, 1))

print("Moyenne des couleurs dans le sol : ", sol_mean)
print("Moyenne des couleurs dans la végétation : ", vegetation_mean)
print("Moyenne des couleurs dans l'eau : ", eau_mean)

cv2.destroyAllWindows()

# Définition des longueurs d'onde approximatives des bandes Sentinel-2 utilisées (en nm)
longueurs_donde = [490, 560, 665, 842]

# Tracer les profils spectraux
plt.figure(figsize=(8, 5))
plt.plot(longueurs_donde, sol_mean, label="Sol", color='green')
plt.plot(longueurs_donde, vegetation_mean, label="Végétation", color='red')
plt.plot(longueurs_donde, eau_mean, label="Eau", color='blue')

# Ajout des légendes et labels
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance moyenne")
plt.title("Profil spectral des différentes classes")
plt.legend()
plt.grid(True)

# Affichage du graphe
plt.show()

# Conversion de l'image en float pour éviter les erreurs de division
image_avant_float = image_avant.astype(np.float32)

# Calcul des distances
def distance_euclidienne(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def spectral_angle_mapper(a, b):
    # Eviter la division par zéro, pour gérer les cas où les vecteurs sont nuls
    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))
    
    if norm_a == 0 or norm_b == 0:
        return np.nan  # Retourne NaN si la norme d'un des vecteurs est zéro
    
    return np.arccos(np.sum(a * b) / (norm_a * norm_b))

# Image de classification (3 canaux pour l'affichage en couleur)
image_euclidienne = np.zeros((dim_avant[0], dim_avant[1], 3), dtype=np.uint8)
image_sam = np.zeros((dim_avant[0], dim_avant[1], 3), dtype=np.uint8)

# Parcours de tous les pixels
for i in range(dim_avant[0]):
    for j in range(dim_avant[1]):
        pixel = image_avant_float[i, j, :]  # Pixel multispectral (4 bandes), maintenant en float

        # Calcul des distances avec les moyennes des classes
        d1_sol = distance_euclidienne(pixel, sol_mean)
        d1_vegetation = distance_euclidienne(pixel, vegetation_mean)
        d1_eau = distance_euclidienne(pixel, eau_mean)

        d2_sol = spectral_angle_mapper(pixel, sol_mean)
        d2_vegetation = spectral_angle_mapper(pixel, vegetation_mean)
        d2_eau = spectral_angle_mapper(pixel, eau_mean)

        # Choix de la classe la plus proche pour Euclidienne
        distances1 = [d1_sol, d1_vegetation, d1_eau]
        classe1 = np.argmin(distances1)  # Index de la classe la plus proche

        # Choix de la classe la plus proche pour SAM
        distances2 = [d2_sol, d2_vegetation, d2_eau]
        classe2 = np.argmin(distances2)  # Index de la classe la plus proche

        # Attribution de couleur (R,G,B) pour Euclidienne
        if classe1 == 0:  # Sol  -> Rouge
            image_euclidienne[i, j] = [255, 0, 0]
        elif classe1 == 1:  # Végétation -> Vert
            image_euclidienne[i, j] = [0, 255, 0]
        elif classe1 == 2:  # Eau -> Bleu
            image_euclidienne[i, j] = [0, 0, 255]
            
        # Attribution de couleur (R,G,B) pour SAM
        if classe2 == 0:  # Sol  -> Rouge
            image_sam[i, j] = [0, 0, 255]
        elif classe2 == 1:  # Végétation -> Vert
            image_sam[i, j] = [0, 255, 0]
        elif classe2 == 2:  # Eau -> Bleu
            image_sam[i, j] = [255, 0, 0]

# Sauvegarde des images classifiées
cv2.imwrite("image_classifiee_euclidienne.png", image_euclidienne)
cv2.imwrite("image_classifiee_sam.png", image_sam)

print("Classification terminée et images sauvegardées !")