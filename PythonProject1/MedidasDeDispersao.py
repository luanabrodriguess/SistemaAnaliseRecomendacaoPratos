import pandas as pd
import numpy as np

# --- Carregar e Preparar os Dados (incluído em cada arquivo para execução independente) ---
try:
    df = pd.read_excel("Dados.xlsx", header=1)
    df = df[['Nota_Atendimento', 'Nota_Geral']].iloc[0:10002]
    print("Dados carregados com sucesso para MedidasDeDispersao.py")

    X = df['Nota_Atendimento']
    y = df['Nota_Geral']

    X = pd.to_numeric(X, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    data = pd.DataFrame({'X': X, 'y': y})
    data_cleaned = data.dropna()

    X_cleaned = data_cleaned['X'].values
    y_cleaned = data_cleaned['y'].values
    print(f"Dados preparados para MedidasDeDispersao.py. {len(data_cleaned)} linhas após a limpeza de NaNs.")

except Exception as e:
    print(f"Erro ao carregar/preparar dados em MedidasDeDispersao.py: {e}")
    X_cleaned = np.array([])
    y_cleaned = np.array([])
    print("Não foi possível calcular medidas de dispersão devido ao erro nos dados.")


# --- Código para Calcular Medidas de Dispersão ---
if len(X_cleaned) > 0 and len(y_cleaned) > 0:
    print("\n### Medidas de Dispersão:")

    print("\nMedidas de Dispersão para Nota_Atendimento:")
    print(f"  Desvio Padrão: {np.std(X_cleaned):.4f}")
    print(f"  Variância: {np.var(X_cleaned):.4f}")
    Q1_X = np.percentile(X_cleaned, 25)
    Q3_X = np.percentile(X_cleaned, 75)
    IQR_X = Q3_X - Q1_X
    print(f"  Intervalo Interquartil (IQR): {IQR_X:.4f}")
    print(f"  Amplitude: {np.ptp(X_cleaned):.4f}")

    print("\nMedidas de Dispersão para Nota_Geral:")
    print(f"  Desvio Padrão: {np.std(y_cleaned):.4f}")
    print(f"  Variância: {np.var(y_cleaned):.4f}")
    Q1_y = np.percentile(y_cleaned, 25)
    Q3_y = np.percentile(y_cleaned, 75)
    IQR_y = Q3_y - Q1_y
    print(f"  Intervalo Interquartil (IQR): {IQR_y:.4f}")
    print(f"  Amplitude: {np.ptp(y_cleaned):.4f}")
else:
     print("\nNão há dados limpos disponíveis para calcular medidas de dispersão.")