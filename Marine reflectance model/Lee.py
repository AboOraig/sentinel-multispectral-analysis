import numpy as np
import matplotlib.pyplot as plt

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

    # Lecture des données d'absorption
    data1 = np.loadtxt("aw.txt")
    aw1 = data1[:, 1]

    data2 = np.loadtxt("aphya.txt")
    aw2 = data2[:, 1]

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

    # Lecture des données de réflectance du fond
    data3 = np.loadtxt("grey_sand.txt")
    re_s = data3[:, 1]
    data4 = np.loadtxt("small_vegetation.txt")
    re_a = data4[:, 1]

    # Calcul des contributions à la réflectance
    r_rsc = r_rsdp * (1 - np.exp(-K * z * ((1 / np.cos(tetaw)) + (D_uc / np.cos(tetav)))))
    phi_f = alpha * re_s + (1 - alpha) * re_a
    r_rsb = (1 / np.pi) * phi_f * np.exp(-K * z * ((1 / np.cos(tetaw)) + (D_ub / np.cos(tetav))))

    # Réflectance totale
    r_rs = r_rsc + r_rsb
    R_rs = (0.52 * r_rs) / (1 - 1.56 * r_rs)

    # Retourne la réflectance totale
    return R_rs

# Données de longueur d'onde
data1 = np.loadtxt("aw.txt")
wl = data1[:, 0]  # Longueur d'onde (nm)

# **Test 1 : Influence de la chlorophylle**
X = [0.01, 0.0001, 0.0001, 100, 1]
plt.figure(1)
for C_chl in [0.01, 0.1, 1, 10]:
    X[0] = C_chl
    R_rs = lee(X, wl)
    plt.plot(wl, R_rs, label=f'C_chl={C_chl}')
plt.title("Influence de la chlorophylle sur la réflectance")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.legend()
plt.show()

# **Test 2 : Influence des particules minérales**
X = [0.0001, 0.0001, 0.0001, 100, 1]
plt.figure(2)
for C_nap in [0.1, 1, 5, 10, 50]:
    X[1] = C_nap
    R_rs = lee(X, wl)
    plt.plot(wl, R_rs, label=f'C_nap={C_nap}')
plt.title("Influence des particules minérales sur la réflectance")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.legend()
plt.show()

# **Test 3 : Influence de la matière organique dissoute**
X = [0.001, 0.0001, 0.0001, 100, 1]
plt.figure(3)
for C_cdom in [0.001, 0.01, 0.1, 1]:
    X[2] = C_cdom
    R_rs = lee(X, wl)
    plt.plot(wl, R_rs, label=f'C_cdom={C_cdom}')
plt.title("Influence de la matière organique dissoute sur la réflectance")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.legend()
plt.show()

# **Test 4 : Influence de la profondeur en eau claire**
X = [0.0001, 0.0001, 0.0001, 100, 1]
plt.figure(4)
for z in [50, 20, 10, 5, 2]:
    X[3] = z
    R_rs = lee(X, wl)
    plt.plot(wl, R_rs, label=f'z={z} m')
plt.title("Influence de la profondeur (eau claire)")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.legend()
plt.show()

# **Test 5 : Influence de la profondeur en eau turbide**
X = [10, 50, 0.1, 100, 1]
plt.figure(5)
for z in [50, 1]:
    X[3] = z
    R_rs = lee(X, wl)
    plt.plot(wl, R_rs, label=f'z={z} m')
plt.title("Influence de la profondeur (eau turbide)")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.legend()
plt.show()

# **Test 6 : Influence du type de fond**
X = [0.0001, 0.0001, 0.0001, 2, 1]  
# Lecture des spectres des fonds
data3 = np.loadtxt("grey_sand.txt")
re_s = data3[:, 1]  # Réflectance du sable
data4 = np.loadtxt("small_vegetation.txt")
re_a = data4[:, 1]  # Réflectance des algues (végétation basse)

# Scénarios de proportions de fond
fonds = [
    (1.0, 0.0, "100% sable"),  # 100% sable
    (0.0, 1.0, "100% algues"),  # 100% algues
    (0.5, 0.5, "50% sable, 50% algues")  # Mixte
]

plt.figure(6)
for alpha_sable, alpha_algue, label in fonds:
    # Modifier la proportion de fond
    X[4] = alpha_sable  # Proportion de sable
    phi_f = alpha_sable * re_s + alpha_algue * re_a
    # Calculer la réflectance
    R_rs = lee(X, wl)
    plt.plot(wl, R_rs, label=label)
plt.title("Influence du type de fond sur la réflectance (eau claire, z=2 m)")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.legend()
plt.show()
