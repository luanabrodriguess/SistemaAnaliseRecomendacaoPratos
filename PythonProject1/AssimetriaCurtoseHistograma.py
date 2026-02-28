import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# --- Carregar e Preparar os Dados (incluído em cada arquivo para execução independente) ---
try:
    df = pd.read_excel("Dados e Estatísticas.xlsx", header=1)
    df = df[['Nota_Atendimento', 'Nota_Geral']].iloc[0:10002]
    print("Dados carregados com sucesso para HistogramaAssimetriaCurtose.py")

    X = df['Nota_Atendimento']
    y = df['Nota_Geral']

    X = pd.to_numeric(X, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    data = pd.DataFrame({'X': X, 'y': y})
    data_cleaned = data.dropna()

    X_cleaned = data_cleaned['X'].values
    y_cleaned = data_cleaned['y'].values
    print(f"Dados preparados para HistogramaAssimetriaCurtose.py. {len(data_cleaned)} linhas após a limpeza de NaNs.")

except Exception as e:
    print(f"Erro ao carregar/preparar dados em HistogramaAssimetriaCurtose.py: {e}")
    X_cleaned = np.array([])
    y_cleaned = np.array([])
    print("Não foi possível gerar histogramas ou calcular assimetria/curtose devido ao erro nos dados.")


# --- Código para Histograma, Assimetria e Curtose ---
if len(X_cleaned) > 0 and len(y_cleaned) > 0:
    # Calcular e exibir assimetria
    print("\n### Assimetria (Skewness):")
    print(f"Assimetria para Nota_Atendimento: {skew(X_cleaned):.4f}")
    print(f"Assimetria para Nota_Geral: {skew(y_cleaned):.4f}")

    # Calcular e exibir curtose
    print("\n### Curtose (Kurtosis):")
    print(f"Curtose para Nota_Atendimento: {kurtosis(X_cleaned, fisher=True):.4f}")
    print(f"Curtose para Nota_Geral: {kurtosis(y_cleaned, fisher=True):.4f}")


    # Gerar histogramas
    print("\n### Histogramas:")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(X_cleaned, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histograma de Nota de Atendimento')
    plt.xlabel('Nota de Atendimento')
    plt.ylabel('Frequência')
    plt.grid(axis='y', alpha=0.75)

    plt.subplot(1, 2, 2)
    plt.hist(y_cleaned, bins=20, color='lightcoral', edgecolor='black')
    plt.title('Histograma de Nota Geral')
    plt.xlabel('Nota Geral')
    plt.ylabel('Frequência')
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.show()
else:
    print("\nNão há dados limpos disponíveis para calcular assimetria/curtose ou gerar histogramas.")