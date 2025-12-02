import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generar_dataset_cry(
    n_puntos=70,
    ruido_gauss=0.08,
    ruido_prop=0.06,
    prob_outlier=0.04,
    outlier_factor=0.25,  # ahora outliers razonables para unidades g/L
    variacion_sist=0.03
):
    """
    Genera datos experimentales sintéticos realistas de producción Cry
    en unidades de g/L, con duración total de 28 h.
    El valor máximo real oscila entre 0 y 1.4 g/L.
    """

    # -------------------------------
    # 1. Eje temporal (0–28 h)
    # -------------------------------
    t = np.linspace(0, 28, n_puntos)

    # -------------------------------
    # 2. Curva biológica real de Cry (sin unidades aún)
    # -------------------------------
    def curva_biologica(t):
        # fase lag (0–4 h)
        lag = np.exp(-((t-3)/2.8)**2) * 0.10

        # log: subida rápida (4–12 h)
        log = 1 / (1 + np.exp(-(t-9)/1.7))

        # pico ~ 14–18 h
        pico = np.exp(-((t-16)/4)**2)

        # declive lento
        declive = np.exp(-(t-18)/10)

        # Combinación para una forma realista
        return (0.15*lag + 0.75*log + pico*declive)

    y_base = curva_biologica(t)

    # Normalizamos al rango 0–1.4 g/L exactamente
    y_base = y_base / np.max(y_base) * 1.4

    # -------------------------------
    # 3. Ruido
    # -------------------------------
    ruido1 = np.random.normal(0, ruido_gauss * 1.4, n_puntos)
    ruido2 = np.random.normal(0, ruido_prop * y_base)

    # -------------------------------
    # 4. Outliers realistas
    # -------------------------------
    outliers_mask = np.random.rand(n_puntos) < prob_outlier
    outliers = np.zeros(n_puntos)

    outliers[outliers_mask] = np.random.normal(
        0,
        outlier_factor,  # variación en g/L
        size=sum(outliers_mask)
    )

    # -------------------------------
    # 5. Variación sistemática
    # -------------------------------
    tendencia = variacion_sist * np.sin(0.25 * t) * y_base

    # -------------------------------
    # 6. Señal final
    # -------------------------------
    y_obs = y_base + ruido1 + ruido2 + outliers + tendencia
    y_obs = np.clip(y_obs, 0, 1.4)  # límites biológicos

    # -------------------------------
    # 7. DataFrame
    # -------------------------------
    df = pd.DataFrame({
        "tiempo_h": t,
        "cry_gL": y_obs,
        "base_biologica": y_base,
        "ruido_gauss": ruido1,
        "ruido_prop": ruido2,
        "outliers": outliers,
        "variacion_sist": tendencia
    })

    # -------------------------------
    # 8. GRAFICA SOLO DATOS
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(t, y_obs, color="black", s=55, label="Datos experimentales")
    plt.xlabel("Tiempo (h)")
    plt.ylabel("Concentración Cry (g/L)")
    plt.title("Producción sintética de Cry (0–1.4 g/L, proceso 28 h)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 9. Guardar CSV
    # -------------------------------
    df.to_csv("dataset_cry_28h.csv", index=False)
    print("Archivo generado: dataset_cry_28h.csv")

    return df

# Ejecutar
df = generar_dataset_cry()

