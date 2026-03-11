# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:36:47 2025

@author: maboorai279
"""

import numpy as np
import matplotlib.pyplot as plt
from lee import lee  # Importation du modèle de réflectance de l'eau
from scipy.optimize import curve_fit  # Importation de la fonction d'optimisation

# Définition des valeurs réelles des paramètres (valeurs de référence)
Xreel = [0.1, 0.1, 0.1, 10, 0.8]  # [C_chl, C_nap, C_cdom, H, alpha]

# Lecture des longueurs d’onde et des coefficients d’absorption de l’eau
wl, aw = np.loadtxt("hico_aw.txt", unpack=True)

# Calcul de la réflectance mesurée à partir des valeurs réelles des paramètres
Rrsmesure = lee(Xreel, wl)  # Simulation de la réflectance spectrale

# Définition des paramètres initiaux pour l’optimisation
X0 = [1.0, 1.0, 0.15, 20.0, 0.5]  # Estimation initiale des paramètres

# Définition des bornes inférieures et supérieures pour chaque paramètre
lb = [0, 0, 0, 0, 0]  # Valeurs minimales autorisées
ub = [10, 10, 1, 50, 1]  # Valeurs maximales autorisées

# Définition de la fonction de coût pour l'optimisation
def lee_optim(wl, C_chl, C_nap, C_cdom, H, alpha):
    """
    Fonction utilisée pour l'ajustement des courbes avec l'optimisation.
    Elle appelle la fonction lee() avec les paramètres optimisés.
    """
    return lee([C_chl, C_nap, C_cdom, H, alpha], wl)

# Ajustement des courbes : optimisation des paramètres pour minimiser l'écart avec Rrsmesure
Xopt, _ = curve_fit(lee_optim, wl, Rrsmesure, p0=X0, bounds=(lb, ub), method='trf')

# Affichage des résultats
plt.figure(1)

# Courbe de la réflectance mesurée (référence)
plt.plot(Rrsmesure, 'g', label='Rrsmesure')

# Courbe de la réflectance simulée avec l’estimation initiale (X0)
plt.plot(lee(X0, wl), 'r', label='Initial guess (X0)')

# Courbe de la réflectance optimisée après l’ajustement des paramètres
plt.plot(lee(Xopt, wl), 'b--', label='Optimized (Xopt)')

# Ajout de la légende et affichage du graphique
plt.xlabel('Longueur ''onde (nm)')
plt.ylabel('Réflectance')
plt.legend()
plt.title('Optimisation des paramètres du modèle de Lee')
plt.show()