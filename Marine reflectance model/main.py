import numpy as np
import matplotlib.pyplot as plt

# Paramètres d'entrée pour les concentrations des composants et autres variables
C_chl = 10                  # Concentration en chlorophylle (mg/m^3)
C_nap = 10                  # Concentration en particules en suspension (g/m^3)
C_cdom = 0.01               # Concentration en matière organique dissoute (g/m^3)
z = 100                     # Profondeur (m)
tetaw = 30 * np.pi / 180    # Angle solaire en radians
tetav = 0 * np.pi / 180     # Angle de visée en radians
alpha = 1.0                 # Proportion de contribution entre différents types de fond

# Constantes spécifiques pour les calculs
S_cdom = 0.0157             # Pente spectrale des matières organiques dissoutes
S_nap = 0.0106              # Pente spectrale des particules en suspension
c_a_nap = 0.0048            # Coefficient d'absorption spécifique des particules en suspension
c_b_bphy = 0.00038          # Coefficient de rétrodiffusion spécifique du phytoplancton
Y_phy = 0.681               # Facteur de pente pour le phytoplancton
c_b_bnap = 0.0054           # Coefficient de rétrodiffusion spécifique des particules en suspension
Y_nap = 0.681               # Facteur de pente pour les particules en suspension

# Lecture des données d'absorption de l'eau pure depuis un fichier texte
data1 = np.loadtxt("aw.txt")
wl = data1[:, 0]            # Longueur d'onde (nm)
aw1 = data1[:, 1]           # Coefficients d'absorption de l'eau pure

# Tracé du spectre d'absorption de l'eau pure
plt.figure(1)
plt.plot(wl, aw1)
plt.title("Absorption de l'eau")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorption")
plt.show()

# Lecture des données d'absorption spécifique du phytoplancton
data2 = np.loadtxt("aphya.txt")
aw2 = data2[:, 1]           # Coefficients d'absorption du phytoplancton

# Tracé du spectre d'absorption du phytoplancton
plt.figure(2)
plt.plot(wl, aw2)
plt.title("Absorption du phytoplancton")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorption")
plt.show()

# Calcul de l'absorption totale
a_phy = C_chl * aw2                                     # Absorption due au phytoplancton
a_nap = C_nap * c_a_nap * np.exp(-S_nap * (wl - 440))   # Absorption due aux particules
a_cdom = C_cdom * np.exp(-S_cdom * (wl - 440))          # Absorption due à la matière organique dissoute
a = aw1 + a_phy + a_nap + a_cdom                        # Absorption totale

# Tracé de l'absorption totale
plt.figure(3)
plt.plot(wl, a)
plt.title("Absorption totale")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorption")
plt.show()

# Calcul de la rétrodiffusion de l'eau pure
b_bw = 0.00144 * (wl / 500)**-4.32      # Coefficients de rétrodiffusion de l'eau pure
plt.figure(4)
plt.plot(wl, b_bw)
plt.title("Rétrodiffusion de l'eau")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Rétrodiffusion")
plt.show()

# Calcul de la rétrodiffusion particulaire
b_bp = (C_chl * c_b_bphy * (542/wl)**Y_phy) + (C_nap * c_b_bnap * (542/wl)**Y_nap)

# Tracé de la rétrodiffusion particulaire
plt.figure(5)
plt.plot(wl, b_bp)
plt.title("Rétrodiffusion particulaire")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Rétrodiffusion")
plt.show()

# Calcul de la rétrodiffusion totale
b_b = b_bw + b_bp
plt.figure(6)
plt.plot(wl, b_b)
plt.title("Rétrodiffusion totale")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Rétrodiffusion")
plt.show()

# Calcul des variables intermédiaires nécessaires à la réflectance
K = a + b_b                                         # Coefficient d'atténuation diffuse
u = b_b / (a + b_b)                                 # Paramètre u
u_p = b_bp / (a + b_b)                              # Paramètre u_p
D_uc = 1.03 * (1 + 2.4 * u)**0.5                    # Paramètre D_uc
D_ub = 1.04 * (1 + 5.4 * u)**0.5                    # Paramètre D_ub
g_p = 0.184 * (1 - 0.602 * np.exp(-3.852 * u_p))    # Facteur de pondération du phytoplancton
g_w = 0.115                                         # Facteur de pondération de l'eau pure

# Calcul de la réflectance sous-marine
r_rsdp = (g_w * b_bw / (a + b_b)) + (g_p * b_bp / (a + b_b))

# Lecture des données de réflectance du fond
data3 = np.loadtxt("grey_sand.txt")         # Sable gris
re_s = data3[:, 1]
data4 = np.loadtxt("small_vegetation.txt")  # Végétation basse
re_a = data4[:, 1]

# Calcul de la réflectance due à la colonne d'eau
r_rsc = r_rsdp * (1 - np.exp(-K * z * ((1 / np.cos(tetaw)) + (D_uc / np.cos(tetav)))))

# Tracé de la réflectance de la colonne d'eau
plt.figure(7)
plt.plot(wl, r_rsc)
plt.title("Réflectance de la colonne d'eau")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.show()

# Contribution du fond à la réflectance
phi_f = alpha * re_s + (1 - alpha) * re_a
r_rsb = (1 / np.pi) * phi_f * np.exp(-K * z * ((1 / np.cos(tetaw)) + (D_ub / np.cos(tetav))))

# Tracé de la réflectance due au fond
plt.figure(8)
plt.plot(wl, r_rsb)
plt.title("Réflectance due au fond")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.show()

# Réflectance totale
r_rs = r_rsc + r_rsb
plt.figure(9)
plt.plot(wl, r_rs)
plt.title("Réflectance totale (sous-marine)")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.show()

# Réflectance de télédétection au-dessus de la surface
R_rs = (0.52 * r_rs) / (1 - 1.56 * r_rs)
plt.figure(10)
plt.plot(wl, R_rs)
plt.title("Réflectance de télédétection")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.show()