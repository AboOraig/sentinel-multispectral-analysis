# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:41:42 2025

@author: maboorai279
"""

import numpy as np
import matplotlib.pyplot as plt

# **Étape 1 : Lecture et visualisation des données hyperspectrales**
# Nom du fichier contenant l'image hyperspectrale HICO
name_file = 'petite_image_HICO_var_26_mai_2013.img'

# **Lecture du fichier binaire**
with open(name_file, 'rb') as Fid:
    C = 400  # Nombre de colonnes (largeur de l'image)
    L = 400  # Nombre de lignes (hauteur de l'image)
    B = 87   # Nombre de bandes spectrales (profondeur spectrale)
    
    # Chargement des données en format uint16 (16 bits non signés)
    donnees = np.fromfile(Fid, dtype=np.uint16, count=C * L * B)

# **Reshape des données pour retrouver la structure 3D de l'image**
CNt = np.reshape(donnees, (C, L, B), order='F')  # Format Fortran (colonne majeure)

# **Correction de l'orientation de l'image**
CNt = np.transpose(CNt, (1, 0, 2))  # Transposition pour remettre l'image dans le bon sens

# **Conversion en luminance TOA (Top of Atmosphere)**
L_TOA = CNt / 50.0  # Normalisation des valeurs

# **Construction d'une image en fausses couleurs**
toto = np.zeros((C, L, 3))  # Création d'une image RGB vide
toto[:, :, 0] = L_TOA[:, :, 41]  # Rouge : Bande 41
toto[:, :, 1] = L_TOA[:, :, 25]  # Vert : Bande 25
toto[:, :, 2] = L_TOA[:, :, 10]  # Bleu : Bande 10

# **Affichage de l'image en fausses couleurs**
plt.figure(1)
plt.imshow((toto * 5).astype(np.uint16))  # Multiplication pour améliorer la visibilité
plt.title('L_TOA - Image en fausses couleurs')
plt.show()

# **Affichage des spectres de pixels caractéristiques**
plt.figure(2)

# **Spectre pour un pixel d'eau claire**
spectre = L_TOA[250, 100, :]  
plt.plot(np.arange(B), spectre, label='Eau claire')

# **Spectre pour un pixel d'eau turbide**
spectre = L_TOA[150, 210, :]  
plt.plot(np.arange(B), spectre, label='Eau turbide')

# **Spectre pour un pixel de végétation**
spectre = L_TOA[100, 50, :]  
plt.plot(np.arange(B), spectre, label='Végétation')

# **Configuration du graphique**
plt.xlabel('Indice de bande spectrale')
plt.ylabel('Luminance TOA')
plt.title('Spectres des pixels sélectionnés')
plt.legend()
plt.show()


# **Étape 2 : Correction de l’éclairement solaire**

# Jour de l’année correspondant à l’acquisition de l’image (26 mai -> jour 146)
j = 146  

# Calcul du facteur de correction de la distance Terre-Soleil (d)
d = (1 + 0.016 * np.cos(2 * np.pi * (j - 3) / 365)) 

# Chargement du fichier contenant l’éclairement solaire extraterrestre F0 (en fonction de la longueur d’onde)
data = np.loadtxt('F0.txt')
wl = data[:, 0]  # Longueurs d'onde (nm)
F0 = data[:, 1]  # Flux solaire extraterrestre (W/m²/µm)

# Angles géométriques (donnés en degrés et convertis en radians)
teta_s = 62.06 * np.pi / 180  # Angle solaire au moment de l’acquisition
teta_v = 16.10 * np.pi / 180  # Angle de visée du capteur

# Initialisation de la matrice de réflectance TOA (même dimension que L_TOA)
ro_TOA = np.zeros(L_TOA.shape)

# Conversion de la luminance TOA en réflectance TOA
for b in range(B):
    ro_TOA[:, :, b] = (np.pi * L_TOA[:, :, b]) / (d * F0[b] * np.cos(teta_s))

# **Étape 3 : Création du masque pour l’eau**

# Seuil de luminance pour isoler l’eau
seuil = 5  

# Création du masque en utilisant la dernière bande spectrale (bande 86)
masque = L_TOA[:, :, 86] < seuil  

# Affichage du masque en niveaux de gris
plt.figure(3)
plt.imshow(masque, cmap='gray')
plt.title('Masque pour l\'eau')
plt.show()


# **Étape 4 : Calcul des transmittances atmosphériques**
# Détermination du coefficient d'atténuation des aérosols (eta)
# Le coefficient eta est calculé à partir des valeurs de transmission à 440 nm et 1020 nm
eta = np.log(0.064 / 0.024) / np.log(440 / 1020)

# Chargement des fichiers de transmittance
to_r = np.loadtxt('to_r.txt')  # Transmittance Rayleigh
to_r = to_r[:, 1]  # On extrait uniquement les valeurs de transmittance
to_g = np.loadtxt('to_g.txt')  # Transmittance due aux gaz atmosphériques
to_g = to_g[:, 1]  # Extraction des valeurs de transmittance

# Calcul de l'atténuation due aux aérosols (to_a)
to_a = 0.024 * (wl / 1020)**eta  # Modélisation de l'effet des aérosols

# Calcul des transmittances atmosphériques
t_s = np.exp(-(0.5 * to_r + to_a) / np.cos(teta_s))  # Transmittance sur le trajet Soleil -> surface
t_v = np.exp(-(0.5 * to_r + to_a) / np.cos(teta_v))  # Transmittance sur le trajet surface -> capteur
t = t_s * t_v  # Transmittance totale en combinant l'aller-retour

# Calcul de la transmittance due aux gaz (tg)
tg_s = np.exp(-to_g / np.cos(teta_s))  # Transmittance des gaz sur le trajet Soleil -> surface
tg_v = np.exp(-to_g / np.cos(teta_v))  # Transmittance des gaz sur le trajet surface -> capteur
tg = tg_s * tg_v  # Transmittance totale des gaz en combinant l'aller-retour

# Affichage de la transmittance atmosphérique en fonction de l'indice de bande
plt.figure(4)
plt.plot(tg)  # Tracé de la transmittance due aux gaz pour chaque bande spectrale
plt.title('Transmittance tg pour chaque bande')
plt.xlabel('Indice de bande')
plt.ylabel('tg')
plt.show()

# **Étape 5 : Correction de Rayleigh**
# Chargement des données de réflectance Rayleigh (ro_r)
ro_r = np.loadtxt('rho_r.txt')
ro_r = ro_r[:, 1]  # Extraction de la colonne des valeurs de réflectance Rayleigh

# Correction de la réflectance TOA en enlevant la contribution de Rayleigh
ro_c = np.zeros(L_TOA.shape)

# Application de la correction de Rayleigh pour chaque bande spectrale
for b in range(B):
    ro_c[:, :, b] = ro_TOA[:, :, b] - tg[b] * ro_r[b]

# Tracé des spectres corrigés pour des pixels représentatifs
plt.figure(5)

# Spectre corrigé pour un pixel d'eau claire
spectre = ro_c[250, 100, :]  
plt.plot(np.arange(B), spectre, label='Eau claire')

# Spectre corrigé pour un pixel d'eau turbide
spectre = ro_c[150, 210, :]  
plt.plot(np.arange(B), spectre, label='Eau turbide')

# Spectre corrigé pour un pixel de végétation
spectre = ro_c[100, 50, :]  
plt.plot(np.arange(B), spectre, label='Végétation')

# Configuration du graphique
plt.xlabel('Indice de bande spectrale')
plt.ylabel('Réflectance corrigée')
plt.title('Spectres corrigés après suppression de Rayleigh')
plt.legend()
plt.show()

# Visualisation de l’image corrigée après suppression de Rayleigh
toto_s = np.zeros((C, L, 3))  # Création d'une image RGB vide
toto_s[:, :, 0] = ro_c[:, :, 41]  # Rouge : Bande 41
toto_s[:, :, 1] = ro_c[:, :, 25]  # Vert : Bande 25
toto_s[:, :, 2] = ro_c[:, :, 10]  # Bleu : Bande 10

# Affichage de l’image corrigée après correction de Rayleigh
plt.figure(6)
plt.imshow(toto_s / np.max(toto_s))  # Normalisation pour mise à l'échelle des valeurs
plt.title('Image corrigée (ro_c)')
plt.show()

# **Étape 6 : Correction des aérosols**
# Sélection des longueurs d’onde pour l’estimation des aérosols
lambda_1 = 776  # nm  -> Première longueur d’onde dans le proche infrarouge
lambda_2 = 862  # nm  -> Deuxième longueur d’onde dans le proche infrarouge

# Détermination des indices des bandes spectrales correspondant à lambda_1 et lambda_2
index_1 = np.argmin(np.abs(wl - lambda_1))  # Trouver l'indice de la bande la plus proche de 776 nm
index_2 = np.argmin(np.abs(wl - lambda_2))  # Trouver l'indice de la bande la plus proche de 862 nm

# Calcul de la réflectance des aérosols (rho_a) dans le proche infrarouge
# Hypothèse : la réflectance de surface en NIR est faible sur l’eau, donc la réflectance mesurée est dominée par les aérosols
rho_a_1 = tg[index_1] * ro_c[:, :, index_1]  # Réflectance des aérosols à lambda_1
rho_a_2 = tg[index_2] * ro_c[:, :, index_2]  # Réflectance des aérosols à lambda_2

# Calcul du facteur epsilon_as et du coefficient c
# epsilon_as est utilisé pour estimer la décroissance spectrale des aérosols
epsilon_as = (rho_a_1 * tg[index_2]) / (rho_a_2 * tg[index_1])

# Le coefficient c est dérivé de l’évolution spectrale des aérosols entre lambda_1 et lambda_2
c = np.log(epsilon_as) / (lambda_2 - lambda_1)

# Estimation de la réflectance des aérosols pour toutes les bandes spectrales
rho_a = np.zeros(ro_c.shape)
for b in range(B):
    rho_a[:, :, b] = rho_a_1 * np.exp(-c * (wl[b] - lambda_1))

# **Étape 7 : **
# Calcul de la réflectance de surface corrigée des aérosols
rho_s = np.zeros(ro_c.shape)
for b in range(B):
    rho_s[:, :, b] = (1 / t[b]) * (ro_c[:, :, b] - tg[b] * rho_a[:, :, b])

# Tracé des spectres corrigés après suppression des aérosols
plt.figure(7)

# Spectre corrigé pour un pixel d'eau claire
spectre = rho_s[250, 100, :]  
plt.plot(np.arange(B), spectre, label='Eau claire')

# Spectre corrigé pour un pixel d'eau turbide
spectre = rho_s[150, 210, :]  
plt.plot(np.arange(B), spectre, label='Eau turbide')

# Spectre corrigé pour un pixel de végétation
spectre = rho_s[100, 50, :]  
plt.plot(np.arange(B), spectre, label='Végétation')

# Configuration du graphique
plt.xlabel('Indice de bande spectrale')
plt.ylabel('Réflectance corrigée de surface (rho_s)')
plt.title('Spectres corrigés des pixels après suppression des aérosols')
plt.legend()
plt.show()

# Visualisation de l’image corrigée après suppression des aérosols
toto_s = np.zeros((C, L, 3))  # Création d'une image RGB vide
toto_s[:, :, 0] = rho_s[:, :, 41]  # Rouge : Bande 41
toto_s[:, :, 1] = rho_s[:, :, 25]  # Vert : Bande 25
toto_s[:, :, 2] = rho_s[:, :, 10]  # Bleu : Bande 10

# Affichage de l’image corrigée après correction des aérosols
plt.figure(8)
plt.imshow(toto_s / np.max(toto_s))  # Normalisation pour mise à l'échelle des valeurs
plt.title('Image corrigée après suppression des aérosols (rho_s)')
plt.show()

# Appliquer le masque sur rho_s
rho_s_water = np.zeros(rho_s.shape)
for b in range(B):
    rho_s_water[:, :, b] = rho_s[:, :, b] * masque

toto_s_water = np.zeros((C, L, 3))
toto_s_water[:, :, 0] = rho_s_water[:, :, 41]  # Bande 42
toto_s_water[:, :, 1] = rho_s_water[:, :, 25]  # Bande 26
toto_s_water[:, :, 2] = rho_s_water[:, :, 10]  # Bande 11

plt.figure(9)
plt.imshow(toto_s_water / np.max(toto_s_water))  # Normalisation
plt.title('Image corrigée pour l\'eau (rho_s)')
plt.show()

np.save("ro_s.npy", {"ro_s": rho_s_water})
np.save('masque.npy', {'masque': masque})

plt.figure(10)

spectre = rho_s_water[250, 100, :]  
plt.plot(np.arange(B), spectre, label='Eau claire')

spectre = rho_s_water[150, 210, :]  
plt.plot(np.arange(B), spectre, label='Eau turbide')

plt.xlabel('Indice de bande spectrale')
plt.ylabel('Réflectance corrigée de surface (rho_s)')
plt.title('Spectres corrigés des pixels d\'eau')
plt.legend()
plt.show()

# **Étape 8**
plt.figure(11)

# Réflectance TOA pour un pixel d'eau claire
spectre_TOA = ro_TOA[250, 100, :]
plt.plot(np.arange(B), spectre_TOA, label='rho_TOA (Eau claire)', linestyle='--')

# Réflectance corrigée pour le même pixel
spectre_s = rho_s[250, 100, :]
plt.plot(np.arange(B), spectre_s, label='rho_s (Eau claire)', linestyle='-')

plt.xlabel('Indice de bande spectrale')
plt.ylabel('Réflectance')
plt.title('Comparaison rho_TOA et rho_s (Eau claire)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(12)

# Réflectance corrigée pour un pixel d'eau claire
spectre_s = rho_s[250, 100, :]
plt.plot(np.arange(B), spectre_s, label='Eau claire')

# Réflectance corrigée pour un pixel d'eau turbide
spectre_s = rho_s[150, 210, :]
plt.plot(np.arange(B), spectre_s, label='Eau turbide')

# Réflectance corrigée pour un pixel de végétation
spectre_s = rho_s[100, 50, :]
plt.plot(np.arange(B), spectre_s, label='Végétation')

plt.xlabel('Indice de bande spectrale')
plt.ylabel('Réflectance corrigée (rho_s)')
plt.title('Spectres corrigés de différents pixels')
plt.legend()
plt.grid(True)
plt.show()
