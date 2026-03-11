# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:41:56 2025

@author: maboorai279
"""

import numpy as np

# Définition de la fonction `lee`
def lee(X, wl):
    C_chl = X[0]
    C_nap = X[1]
    C_cdom = X[2]
    z = X[3]
    alpha = X[4]
    
    tetaw = 30 * np.pi / 180  # Angle solaire
    tetav = 0 * np.pi / 180   # Angle de visée

    # Constantes spécifiques
    S_cdom = 0.0157
    S_nap = 0.0106
    c_a_nap = 0.0048
    c_b_bphy = 0.00038
    Y_phy = 0.681
    c_b_bnap = 0.0054
    Y_nap = 0.681

    wl, aw1 = np.loadtxt("hico_aw.txt", unpack=True)
    wl, aw2 = np.loadtxt("hico_aphya.txt", unpack=True)
    wl, re_s = np.loadtxt("hico_grey_sand.txt", unpack=True)
    wl, re_a = np.loadtxt("hico_small_vegetation.txt", unpack=True)

    # Calcul de l'absorption totale
    a_phy = C_chl * aw2
    a_nap = C_nap * c_a_nap * np.exp(-S_nap * (wl - 440))
    a_cdom = C_cdom * np.exp(-S_cdom * (wl - 440))
    a = aw1 + a_phy + a_nap + a_cdom

    # Calcul des coefficients de rétrodiffusion
    b_bw = 0.00144 * (wl / 500)**-4.32
    b_bp = C_chl * c_b_bphy * (542/wl)**Y_phy + C_nap * c_b_bnap * (542/wl)**Y_nap
    b_b = b_bw + b_bp

    # Calcul des variables intermédiaires
    K = a + b_b
    u = b_b / (a + b_b)
    u_p = b_bp / (a + b_b)
    D_uc = 1.03 * (1 + 2.4 * u)**0.5
    D_ub = 1.04 * (1 + 5.4 * u)**0.5
    g_p = 0.184 * (1 - 0.602 * np.exp(-3.852 * u_p))
    g_w = 0.115
    r_rsdp = (g_w * b_bw / (a + b_b)) + (g_p * b_bp / (a + b_b))


    # Calcul des contributions à la réflectance
    r_rsc = r_rsdp * (1 - np.exp(-K * z * ((1 / np.cos(tetaw)) + (D_uc / np.cos(tetav)))))
    phi_f = alpha * re_s + (1 - alpha) * re_a
    r_rsb = (1 / np.pi) * phi_f * np.exp(-K * z * ((1 / np.cos(tetaw)) + (D_ub / np.cos(tetav))))

    # Réflectance totale
    r_rs = r_rsc + r_rsb
    R_rs = (0.52 * r_rs) / (1 - 1.56 * r_rs)

    # Retourne la réflectance totale
    return R_rs
