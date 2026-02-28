import pandas as pd
import numpy as np

try:
    df = pd.read_excel("Dados e Estatísticas.xlsx", header=1)
    df = df[['Nota_Atendimento', 'Nota_Geral']].iloc[0:10002]
    print("Dados carregados com sucesso para Percentil.py")

    X = df['Nota_Atendimento']
    y = df['Nota_Geral']

    X = pd.to_numeric(X, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    data = pd.DataFrame({'X': X, 'y': y})
    data_cleaned = data.dropna()

    X_cleaned = data_cleaned['X'].values
    y_cleaned = data_cleaned['y'].values
    print(f"Dados preparados para Percentil.py. {len(data_cleaned)} linhas após a limpeza de NaNs.")

except Exception as e:
    print(f"Erro ao carregar/preparar dados em Percentil.py: {e}")
    X_cleaned = np.array([])
    y_cleaned = np.array([])
    print("Não foi possível realizar o cálculo de percentis devido ao erro nos dados.")

if len(X_cleaned) > 0 and len(y_cleaned) > 0:
    print("\n### Percentis:")
    print("Percentis para Nota_Atendimento:")
    print(np.percentile(X_cleaned, [25, 50, 75, 90, 95]))

    print("\nPercentis para Nota_Geral:")
    print(np.percentile(y_cleaned, [25, 50, 75, 90, 95]))
else:
    print("\nNão há dados limpos disponíveis para calcular percentis.")