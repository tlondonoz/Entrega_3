import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. PARÁMETROS (Columna F1)
# ===========================
X_max = 5.58
PHB_max = 0.61
DPA_max = 0.17
Cry_max = 0.19

# Tasas (1/h)
mu_max = 0.78
mu_maxp = 1.07
mu_maxd = 0.66
mu_maxc = 0.20

# Tiempos (h)
tc = 4.54
tcp = 8.46
tcd = 9
tcc = 14.88

# Rendimientos
Y_DPA_X = 0.025
Y_DPA_PHB = 1.02
Y_Cry_PHB = 0.19

# ===========================
# 2. CÁLCULOS (Modelo Gompertz)
# ===========================
def gompertz(t, A_max, mu, t_lag):
    return A_max * np.exp(-np.exp(-mu * (t - t_lag)))

t = np.linspace(0, 30, 1000)

# Productos finales
Cry = gompertz(t, Cry_max, mu_maxc, tcc)
DPA = gompertz(t, DPA_max, mu_maxd, tcd)

# PHB (Producción - Consumo)
PHB_prod = gompertz(t, PHB_max, mu_maxp, tcp)
consumption_DPA = DPA / Y_DPA_PHB
consumption_Cry = Cry / Y_Cry_PHB
PHB = PHB_prod - consumption_DPA - consumption_Cry
PHB[PHB < 0] = 0

# Biomasa (Crecimiento - Lisis)
X_prod = gompertz(t, X_max, mu_max, tc)
loss_to_spores = DPA / Y_DPA_X
X = X_prod - loss_to_spores
X[X < 0] = 0


# 3. GRAFICACIÓN CON ZOOM
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

#PANEL 1: Panorama General (Domina la Biomasa)
ax1.plot(t, X, label='Biomasa (X)', color='#1f77b4', linewidth=2.5)
ax1.plot(t, PHB, label='PHB', color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.plot(t, DPA, label='Espora (DPA)', color='#2ca02c', linewidth=1.5, alpha=0.7)
ax1.plot(t, Cry, label='Toxina (Cry)', color='#d62728', linewidth=1.5, alpha=0.7)

ax1.set_title('Panorama General (Dominado por Biomasa)', fontsize=14)
ax1.set_ylabel('Conc. (g/L)', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axvline(tcd, color='gray', linestyle=':', alpha=0.5)

# PANEL 2: ZOOM a los Productos (Interacción PHB/Cry/DPA)
# NO graficala biomasa aquí para que la escala se ajuste a los pequeños
ax2.plot(t, PHB, label='PHB (Reserva)', color='#ff7f0e', linewidth=2.5, linestyle='--')
ax2.plot(t, DPA, label='Espora (DPA)', color='#2ca02c', linewidth=2.5)
ax2.plot(t, Cry, label='Toxina (Cry)', color='#d62728', linewidth=2.5)

# Marcar eventos críticos
ax2.axvline(tcp, color='#ff7f0e', linestyle=':', alpha=0.5, label=f'Inicio PHB ({tcp}h)')
ax2.axvline(tcd, color='#2ca02c', linestyle=':', alpha=0.5, label=f'Inicio Esporas ({tcd}h)')

ax2.set_title('ZOOM: Dinámica de Productos (Consumo de PHB)', fontsize=14)
ax2.set_xlabel('Tiempo (h)', fontsize=12)
ax2.set_ylabel('Conc. (g/L)', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)


ax2.fill_between(t, PHB, color='#ff7f0e', alpha=0.1)

plt.tight_layout()
plt.show()