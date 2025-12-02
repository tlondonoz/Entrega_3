import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import fisher_exact
import warnings
import os

warnings.filterwarnings('ignore')

# Configuración de archivos
INPUT_FILE = r"C:\Users\tomas\Desktop\BLAST\RESULTADOS\Entrega3\datos.xlsx"
OUTPUT_DIR = r"C:\Users\tomas\Desktop\BLAST\RESULTADOS\Entrega3"

# Diccionario de mapeo de metabolitos a rutas
diccionario_rutas = {
    'alanine': 'Alanine, aspartate and glutamate metabolism',
    'arginine': 'Arginine and proline metabolism',
    'aspartic acid': 'Alanine, aspartate and glutamate metabolism',
    'cysteine': 'Cysteine and methionine metabolism',
    'glutamic acid': 'Alanine, aspartate and glutamate metabolism',
    'glutamine': 'Alanine, aspartate and glutamate metabolism',
    'glycine': 'Glycine, serine and threonine metabolism',
    'isoleucine': 'Valine, leucine and isoleucine biosynthesis',
    'leucine': 'Valine, leucine and isoleucine biosynthesis',
    'lysine': 'Lysine biosynthesis',
    'methionine': 'Cysteine and methionine metabolism',
    'phenylalanine': 'Phenylalanine metabolism',
    'proline': 'Arginine and proline metabolism',
    'serine': 'Glycine, serine and threonine metabolism',
    'threonine': 'Glycine, serine and threonine metabolism',
    'tryptophan': 'Tryptophan metabolism',
    'tyrosine': 'Tyrosine metabolism',
    'valine': 'Valine, leucine and isoleucine biosynthesis',
    'glutathione': 'Glutathione metabolism',
    'gsh': 'Glutathione metabolism',
    'putrescine': 'Arginine and proline metabolism',
    'spermidine': 'Arginine and proline metabolism',
    'spermine': 'Arginine and proline metabolism',
    'adenine': 'Purine metabolism',
    'adenosine': 'Purine metabolism',
    'guanine': 'Purine metabolism',
    'guanosine': 'Purine metabolism',
    'xanthine': 'Purine metabolism',
    'hypoxanthine': 'Purine metabolism',
    'uric acid': 'Purine metabolism',
    'allantoin': 'Purine metabolism',
    'inosine': 'Purine metabolism',
    'cytosine': 'Pyrimidine metabolism',
    'cytidine': 'Pyrimidine metabolism',
    'thymine': 'Pyrimidine metabolism',
    'uracil': 'Pyrimidine metabolism',
    'uridine': 'Pyrimidine metabolism',
    'citric acid': 'Citrate cycle (TCA cycle)',
    'succinic acid': 'Citrate cycle (TCA cycle)',
    'fumaric acid': 'Citrate cycle (TCA cycle)',
    'malic acid': 'Citrate cycle (TCA cycle)',
    'pyruvic acid': 'Pyruvate metabolism',
    'glucose': 'Starch and sucrose metabolism',
    'sucrose': 'Starch and sucrose metabolism',
    'maltose': 'Starch and sucrose metabolism',
    'galactose': 'Galactose metabolism',
    'cinnamic acid': 'Phenylpropanoid biosynthesis',
    'coumaric acid': 'Phenylpropanoid biosynthesis',
    'caffeic acid': 'Phenylpropanoid biosynthesis',
    'ferulic acid': 'Phenylpropanoid biosynthesis',
    'catechin': 'Flavonoid biosynthesis',
    'epicatechin': 'Flavonoid biosynthesis',
    'naringin': 'Flavonoid biosynthesis',
    'apigenin': 'Flavonoid biosynthesis',
    'luteolin': 'Flavonoid biosynthesis',
    'kaempferide': 'Flavonoid biosynthesis',
    'quercitrin': 'Flavonoid biosynthesis',
    'rutin': 'Flavonoid biosynthesis',
    'riboflavin': 'Riboflavin metabolism',
    'niacin': 'Nicotinate and nicotinamide metabolism',
    'pyridoxine': 'Vitamin B6 metabolism',
    'thiamine': 'Thiamine metabolism',
    'ascorbic acid': 'Ascorbate and aldarate metabolism',
    'folic acid': 'Folate biosynthesis',
}

def buscar_ruta(nombre):
    nombre = str(nombre).lower().strip()
    for llave in diccionario_rutas:
        if llave in nombre:
            return diccionario_rutas[llave]
    return "Sin Mapa"

# 1. Carga y Normalización
if not os.path.exists(INPUT_FILE):
    print(f"Error: Archivo no encontrado en {INPUT_FILE}")
    exit()

df = pd.read_excel(INPUT_FILE, header=1)
df.columns = df.columns.str.strip()

mapa_cols = {'Cell Type': 'CellType', 'IDs': 'SampleID', 'Treatmen': 'Treatment'}
df.rename(columns=mapa_cols, inplace=True)
if 'Cell type' in df.columns: df.rename(columns={'Cell type': 'CellType'}, inplace=True)

cols_meta = ['SampleID', 'CellType', 'Treatment', 'Time']
mets = [c for c in df.columns if c not in cols_meta]

for col in mets:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

if df['Time'].dtype == object:
    df['Time'] = df['Time'].astype(str).str.replace('m', '', regex=False).astype(int)
df['Treatment'] = df['Treatment'].map({'C': 'Control', 'T': 'HCO3'}).fillna(df['Treatment'])

df_log = df.copy()
medianas = df_log[mets].median(axis=1).replace(0, 1)
df_log[mets] = df_log[mets].div(medianas, axis=0)
df_log[mets] = np.log2(df_log[mets] + 1)

df_log.to_excel(os.path.join(OUTPUT_DIR, "1_Datos_Normalizados.xlsx"), index=False)

# 2. ANOVA Two-Way
resultados_anova = []
celulas = ['GC', 'MC']

for celula in celulas:
    subset = df_log[df_log['CellType'] == celula].copy()
    
    for met in mets:
        temp = subset[['Treatment', 'Time']].copy()
        temp['Intensidad'] = subset[met]
        temp = temp.dropna()
        
        try:
            model = ols('Intensidad ~ C(Treatment) + C(Time) + C(Treatment):C(Time)', data=temp).fit()
            aov = sm.stats.anova_lm(model, typ=2)
            
            p_treat = aov.loc['C(Treatment)', 'PR(>F)']
            try:
                p_inter = aov.loc['C(Treatment):C(Time)', 'PR(>F)']
            except:
                p_inter = 1.0
            
            if p_treat < 0.05 or p_inter < 0.05:
                resultados_anova.append({
                    'CellType': celula,
                    'Metabolito': met,
                    'P_Tratamiento': p_treat,
                    'P_Interaccion': p_inter
                })
        except:
            continue

df_anova = pd.DataFrame(resultados_anova)
df_anova.to_excel(os.path.join(OUTPUT_DIR, "2_Filtro_Estadistico_ANOVA.xlsx"), index=False)

# 3. Dirección y Fold Change
lista_fisher = []
tiempos = sorted([t for t in df['Time'].unique() if t > 0])

for index, fila in df_anova.iterrows():
    cel = fila['CellType']
    met = fila['Metabolito']
    
    for t in tiempos:
        sub = df[(df['CellType'] == cel) & (df['Time'] == t)]
        c_mean = sub[sub['Treatment'] == 'Control'][met].mean()
        t_mean = sub[sub['Treatment'] == 'HCO3'][met].mean()
        
        if c_mean == 0: continue
        fc = t_mean / c_mean
        
        direction = "Sin Cambio"
        if fc > 1.2: direction = "Increase"
        elif fc < 0.8: direction = "Decrease"
        
        if direction != "Sin Cambio":
            lista_fisher.append({
                'CellType': cel,
                'Time': t,
                'Metabolito': met,
                'Direccion': direction,
                'Ruta': buscar_ruta(met)
            })

df_fisher_input = pd.DataFrame(lista_fisher)
df_fisher_input.to_excel(os.path.join(OUTPUT_DIR, "3_Lista_Metabolitos_Significativos.xlsx"), index=False)

# 4. Fisher Exact Test
resultados_finales = []
rutas_unicas = df_fisher_input[df_fisher_input['Ruta'] != 'Sin Mapa']['Ruta'].unique()

for cel in celulas:
    for t in tiempos:
        subset = df_fisher_input[(df_fisher_input['CellType'] == cel) & (df_fisher_input['Time'] == t)]
        
        for dir in ['Increase', 'Decrease']:
            grupo = subset[subset['Direccion'] == dir]
            if grupo.empty: continue
            
            total_sig = len(grupo[grupo['Ruta'] != 'Sin Mapa'])
            
            for ruta in rutas_unicas:
                hits = len(grupo[grupo['Ruta'] == ruta])
                if hits == 0: continue
                
                hits_out = total_sig - hits
                
                total_ruta = len(df_fisher_input[df_fisher_input['Ruta'] == ruta]['Metabolito'].unique())
                resto_in = total_ruta - hits
                if resto_in < 0: resto_in = 0
                
                total_map = len(df_fisher_input[df_fisher_input['Ruta'] != 'Sin Mapa']['Metabolito'].unique())
                resto_out = total_map - total_ruta - hits_out
                if resto_out < 0: resto_out = 0
                
                table = [[hits, hits_out], [resto_in, resto_out]]
                odds, p_val = fisher_exact(table, alternative='greater')
                
                if p_val < 0.1:
                    resultados_finales.append({
                        'Célula': cel,
                        'Tiempo (min)': t,
                        'Dirección': dir,
                        'Ruta Metabólica': ruta,
                        'Hits': hits,
                        'P-Value (Fisher)': p_val
                    })

df_final = pd.DataFrame(resultados_finales)
if not df_final.empty:
    df_final.sort_values(['Célula', 'Tiempo (min)', 'Dirección', 'P-Value (Fisher)'], inplace=True)
    df_final.to_excel(os.path.join(OUTPUT_DIR, "4_Resultado_Final_S7.xlsx"), index=False)
    print("Proceso completado. Archivos generados.")
else:
    print("Proceso completado. No se generararon archivos.")