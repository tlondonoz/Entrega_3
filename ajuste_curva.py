import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# Modelo P/Q
# -------------------------
def pq_model(t, Pmax, tmax, a):
    t = np.array(t)
    t = np.where(t == 0, 1e-6, t)
    return Pmax * (t / tmax)**a * np.exp(a * (1 - t / tmax))

# -------------------------
# Cargar dataset real
# -------------------------
ruta = r"C:\Users\tomas\Desktop\BLAST\RESULTADOS\dataset_cry_28h.csv"
df = pd.read_csv(ruta)

# Asegurar que las columnas tengan el nombre correcto
# (si tus columnas reales son tiempo_h y cry_gL, usa esos nombres)
col_t = "tiempo_h"
col_y = "cry_gL"

t = df[col_t].values
y = df[col_y].values

# -------------------------
# Detectar outliers con MAD-Z
# -------------------------
def detectar_outliers_MAD(y, z_thresh=3.5):
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad < 1e-8:
        return np.zeros_like(y, dtype=bool)
    z = 0.6745 * (y - med) / mad
    return np.abs(z) > z_thresh

outlier_mask = detectar_outliers_MAD(y)

# Filtrar datos
t_clean = t[~outlier_mask]
y_clean = y[~outlier_mask]

# -------------------------
# Ajuste de curva PQ
# -------------------------
p0 = [y_clean.max(), t_clean[np.argmax(y_clean)], 3]  # estimación inicial

params, cov = curve_fit(
    pq_model,
    t_clean,
    y_clean,
    p0=p0,
    bounds=([0, 1, 0.5], [2.0, 30.0, 10])
)

Pmax_hat, tmax_hat, a_hat = params

print("AJUSTE DEL MODELO P/Q")
print(f"Pmax = {Pmax_hat:.4f} g/L")
print(f"tmax = {tmax_hat:.3f} h")
print(f"a = {a_hat:.3f}")

# -------------------------
# Curva ajustada fina
# -------------------------
t_fine = np.linspace(t.min(), t.max(), 300)
y_fit = pq_model(t_fine, *params)

# -------------------------
# Gráfica final
# -------------------------
plt.figure(figsize=(10,5))

# Datos reales usados
plt.scatter(t_clean, y_clean, color="black", s=45, label="Datos usados")

# Outliers detectados
plt.scatter(t[outlier_mask], y[outlier_mask], color="red", s=70, marker="x", label="Outliers eliminados")

# Curva ajustada
plt.plot(t_fine, y_fit, color="orange", lw=2.5, label="Ajuste P/Q")

plt.xlabel("Tiempo (h)")
plt.ylabel("Concentración Cry (g/L)")
plt.title("Ajuste del modelo P/Q con eliminación de outliers")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
