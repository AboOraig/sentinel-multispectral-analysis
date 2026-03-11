# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:39:23 2025

@author: mbinsnka280
"""

import numpy as np
import matplotlib.pyplot as plt
from lee import lee
from scipy.optimize import curve_fit

# Longueurs d'onde des bandes HICO
wl_hico = np.linspace(404.08, 896.688, 87)  # 87 bandes avec un pas de 5.728 nm

# Lecture du fichier hico_aw.txt
wl, aw = np.loadtxt('hico_aw.txt', unpack=True)

# Paramètres réels pour générer Rrsmesure
Xreel = [0.1, 0.1, 0.1, 10, 0.8]  # Valeurs réelles des paramètres
Rrsmesure = lee(Xreel, wl)  # Réflectance spectrale mesurée simulée

# Initial guess (X0), bornes et configuration de l'optimisation
X0 = [1.0, 1.0, 0.15, 20.0, 0.5]  # Estimation initiale
lb = [0, 0, 0, 0, 0]  # Bornes inférieures
ub = [10, 10, 1, 50, 1]  # Bornes supérieures

# Définir la fonction optimisée pour lsqcurvefit
def lee_optim(wl, C_chl, C_nap, C_cdom, H, alpha):
    return lee([C_chl, C_nap, C_cdom, H, alpha], wl)

# Optimisation avec curve_fit
Xopt, _ = curve_fit(lee_optim, wl, Rrsmesure, p0=X0, bounds=(lb, ub), method='trf')

# Charger le fichier .npy contenant l'image HICO en réflectance
data = np.load("ro_s.npy", allow_pickle=True).item()  # allow_pickle=True pour charger les objets

# Accéder à l'image HICO en réflectance
ro_s = data["ro_s"]

# Récupérer la taille de l'image (Lignes, Colonnes, Bandes)
L, C, B = ro_s.shape
print("Dimensions de l'image HICO:", L, C, B)

data = np.load("masque.npy", allow_pickle=True).item()

# Charger le masque eau
masque = data["masque"]

# Initialiser la matrice cartographique pour les paramètres X optimisés
carto = np.zeros((L, C, 6))  #  pour les 5 paramètres optimisés

# Balayer l'image en ligne et en colonne
for i in range(L):
    for j in range(C):
        # Vérifier si le pixel appartient à l'eau
        if masque[i, j] != 0:  # 1 signifie que le pixel appartient à l'eau
            Rrs = ro_s[i, j, :]
            
            # Optimisation des paramètres pour ce pixel
            Xopt_pixel, _ = curve_fit(lee_optim, wl, Rrs, p0=X0, bounds=(lb, ub), method='trf')
            
            # Tracer la réflectance mesurée et optimisée
            plt.figure()
            plt.plot(Rrs, label="Réflectance mesurée", color='g')
            plt.plot(lee(Xopt_pixel, wl), label="Réflectance optimisée", linestyle='--', color='b')
            
            # Ajouter titre et légende
            plt.title(f"Optimisation du spectre pour le pixel ({i}, {j})")
            plt.xlabel("Indice de bande spectrale")
            plt.ylabel("Réflectance")
            plt.legend()
            plt.grid(True)
            
            # Pause pour visualisation
            plt.pause(0.01)

            # Stocker les paramètres optimisés pour ce pixel
            # carto[i, j, :] = Xopt_pixel
            carto[i, j, 0] = Xopt_pixel[0]  # Chlorophylle
            carto[i, j, 1] = Xopt_pixel[1]  # NAP
            carto[i, j, 2] = Xopt_pixel[2]  # CDOM
            carto[i, j, 3] = Xopt_pixel[3]  # Profondeur
            carto[i, j, 4] = Xopt_pixel[4]  # Fraction du fond
            carto[i, j, 5] = 1 - Xopt_pixel[4]  # Complément du fond

# Sauvegarder les cartes
np.save("carto.npy", carto)
carto = np.load("carto.npy")

# Visualization of the optimized parameters (chl, nap, cdom, etc.)
fig, axes = plt.subplots(3, 2, figsize=(10, 10))  # Crée une figure avec une grille de 3x2 sous-graphes

# Liste des paramètres à afficher
parameters = ['chl', 'nap', 'cdom', 'Z', 'sable','alpha']

# Affichage de chaque carte paramétrique
for i, ax in enumerate(axes.flat):  # Parcours des axes des sous-graphes
    im = ax.imshow(carto[:, :, i], cmap='jet')  # Affiche la carte sous forme d’image
    ax.set_title(parameters[i])  # Ajoute le titre correspondant au paramètre affiché
    fig.colorbar(im, ax=ax)  # Ajoute une barre de couleur pour indiquer les valeurs
    ax.set_xticks([])  # Supprime les axes inutiles
    ax.set_yticks([])

plt.tight_layout()  # Ajuste l'affichage des sous-graphes pour éviter les chevauchements
plt.show()  # Affiche la figure finale


'''
name_file = 'petite_carto_HICO_var_26_mai_2013.img'
C = 400
L = 400
B = 6
with open(name_file, 'rb') as Fid:
    donnees = np.fromfile(Fid, dtype = np.float32, count = C*L*B)

carto_t = np.rashape(donnees, (C, L, B), order = 'F')
carto = np.zeros((C, L, B))

for i in range(6):
    carto[:, :, i] = carto_t[:, :, i].T
    
fig, axes = plt.subplots(3, 2, figsize=(10, 10)) 
parameters = ['chl', 'nap', 'cdom', 'Z', 'sable','alpha']

for i, ax in enumerate(axes.flat):  # Parcours des axes des sous-graphes
    im = ax.imshow(carto[:, :, i], cmap='jet')  # Affiche la carte sous forme d’image
    ax.set_title(parameters[i])  # Ajoute le titre correspondant au paramètre affiché
    fig.colorbar(im, ax=ax)  # Ajoute une barre de couleur pour indiquer les valeurs

plt.tight_layout()  # Ajuste l'affichage des sous-graphes pour éviter les chevauchements
plt.show()  # Affiche la figure finale
'''