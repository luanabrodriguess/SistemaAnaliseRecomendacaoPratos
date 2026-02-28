# ====================================================================
# SEÇÃO 1: IMPORTS E PREPARAÇÃO DE DADOS
# ====================================================================

import pandas as pd
import numpy as np
# Bibliotecas de ML (Scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, silhouette_score, recall_score, precision_score
# Para o Hierarchical Clustering (Dendrograma)
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# 1. Carregar e Limpar Dados
try:
    nome_arquivo = "Dados.xlsx"

    # CORREÇÃO DEFINITIVA:
    # 1. Usando "Dados.xlsx" e pd.read_excel (conforme seu último feedback)
    # 2. Adicionando nrows=10000 para ler apenas as linhas de DADOS, ignorando as linhas de estatísticas no final
    df = pd.read_excel(nome_arquivo, header=1, nrows=10000)

    # Selecionar as colunas numéricas (Features) relevantes para ML
    features_ml = ['Tempo_Espera_Min', 'Preco_Prato_BRL', 'Nota_Atendimento', 'Nota_Geral']

    # Verificação de colunas para garantir que o cabeçalho foi lido corretamente
    if not all(feature in df.columns for feature in features_ml):
        raise ValueError("O DataFrame não contém todas as colunas esperadas. Verifique o nome das colunas na planilha.")

    df_ml = df[features_ml].copy()
    df_ml = df_ml.dropna()  # Limpeza de linhas com valores ausentes (NaNs)

    print(f"Dados carregados com sucesso usando o arquivo: {nome_arquivo}")
    print(f"Dados ML carregados e limpos: {len(df_ml)} linhas.")

except FileNotFoundError:
    print(f"Erro fatal: Arquivo '{nome_arquivo}' não encontrado.")
    print(f"Verifique se o arquivo Excel/Planilha está na mesma pasta do script Python.")
    exit()
except ValueError as e:
    # Este erro agora só deve ocorrer se houver dados não-numéricos DENTRO das 10000 linhas,
    # o que indica que a limpeza está falhando.
    print(f"Erro no cabeçalho ou estrutura: {e}")
    exit()
except Exception as e:
    print(f"Erro inesperado ao carregar dados: {e}")
    exit()

# Separar Features (X) para ML
X_all = df_ml[['Tempo_Espera_Min', 'Preco_Prato_BRL', 'Nota_Atendimento', 'Nota_Geral']]

# Aplicar Escala (Standardization) - Essencial para K-Means, KNN e Redes Neurais
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_all.columns)

# ====================================================================
# SEÇÃO 2: MACHINE LEARNING NÃO SUPERVISIONADO (CLUSTERING - ITEM 5)
# ====================================================================

print("\n\n### 5.1. K-Means Clustering (Item 5) ###")

# Usar as features escalonadas (Nota, Preço e Tempo)
X_cluster = X_scaled_df[['Preco_Prato_BRL', 'Nota_Geral', 'Tempo_Espera_Min']]

# Método do Cotovelo (para achar o K ideal) - Item B: Gráfico
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo para K-Means')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.show()

# Rodar K-Means com K=4 (Valor sugerido, mas ajuste de acordo com o gráfico)
K_FINAL = 4
kmeans_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_cluster)
df_ml['Cluster'] = clusters

# Métrica de Qualidade (Item A)
sil_score = silhouette_score(X_cluster, clusters)
print(f"Métrica de Qualidade (Coeficiente de Silhueta): {sil_score:.4f}")

# Gráfico de Dispersão dos Clusters (Item B)
plt.figure(figsize=(8, 6))
plt.scatter(df_ml['Preco_Prato_BRL'], df_ml['Nota_Geral'], c=df_ml['Cluster'], cmap='jet', marker='o', alpha=0.6)
plt.title(f'K-Means Clusters (K={K_FINAL}) - Preço vs Nota')
plt.xlabel('Preço do Prato (BRL)')
plt.ylabel('Nota Geral')
plt.show()

print("\n\n### 5.3. Hierarchical Clustering e Dendrograma (Item 5) ###")
# Dendrograma (Item B: Gráfico)
Z = linkage(X_cluster, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='lastp', p=10, show_leaf_counts=True)
plt.title('Dendrograma (Hierarchical Clustering)')
plt.xlabel('Tamanho do Cluster')
plt.ylabel('Distância')
plt.show()

# ====================================================================
# SEÇÃO 3: MACHINE LEARNING SUPERVISIONADO (CLASSIFICAÇÃO - ITEM 6)
# ====================================================================

print("\n\n### 6. Aprendizado Supervisionado: Classificação (Item 6) ###")

# 1. Criar a Variável Alvo Y (Classificação Binária)
# Y = 1 (Sucesso) se Nota_Geral >= 4.5. 0 (Fracasso) caso contrário.
Y = np.where(df_ml['Nota_Geral'] >= 4.5, 1, 0)

# Features (X) - Variáveis que predizem o sucesso
X_features = df_ml[['Tempo_Espera_Min', 'Preco_Prato_BRL', 'Nota_Atendimento']]

# Separar dados de Treino e Teste
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.3, random_state=42)

# Normalizar/Escalar X de Treino e Teste (necessário para KNN e Redes Neurais)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Definição e Treinamento dos 4 Modelos (Requisito Item 6)
modelos = {
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forests": RandomForestClassifier(random_state=42, n_estimators=100),
    "KNN K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Rede Neural (MLP)": MLPClassifier(random_state=42, max_iter=100)
}

resultados = {}

for nome, modelo in modelos.items():
    # Usa dados escalados para modelos sensíveis à escala, e dados brutos para Árvore/RF
    if nome in ["KNN K-Nearest Neighbors", "Rede Neural (MLP)"]:
        modelo.fit(X_train_scaled, Y_train)
        Y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, Y_train)
        Y_pred = modelo.predict(X_test)

    # Avaliação (Item A: Acurácia, F1-Score, Precisão, Recall)
    resultados[nome] = {
        'Acurácia': accuracy_score(Y_test, Y_pred),
        'F1-Score': f1_score(Y_test, Y_pred, zero_division=0),
        'Precisão': precision_score(Y_test, Y_pred, zero_division=0),
        'Recall': recall_score(Y_test, Y_pred, zero_division=0),
        'Matriz_Confusao': confusion_matrix(Y_test, Y_pred)
    }
    print(f"\n--- Resultados para {nome} (Item A) ---")
    print(f"Acurácia: {resultados[nome]['Acurácia']:.4f} | F1-Score: {resultados[nome]['F1-Score']:.4f}")

# 3. Geração de Gráficos e Tabela de Comparação (Itens B e C)

# Tabela de Comparação (para ser usada no Relatório) - Item C
df_resultados = pd.DataFrame(resultados).T[['Acurácia', 'F1-Score', 'Precisão', 'Recall']]
print("\n### Tabela de Comparação de Classificadores (Item C) ###")
print(df_resultados.sort_values(by='Acurácia', ascending=False))

# Gráfico da Árvore de Decisão (Item B)
plt.figure(figsize=(20, 10))
plot_tree(modelos["Árvore de Decisão"], filled=True,
          feature_names=X_features.columns.tolist(),
          class_names=['Fracasso (0)', 'Sucesso (1)'],
          fontsize=10)
plt.title('Árvore de Decisão para Previsão de Sucesso do Prato (Item B)')
plt.show()
print("\n\n### NOVO: Gráfico de Importância de Features (Random Forests) ###")

# Obter o modelo treinado de Random Forest
rf_model = modelos["Random Forests"]

# Extrair a importância das features
importances = rf_model.feature_importances_
feature_names = X_features.columns.tolist()

# Criar um DataFrame para facilitar a ordenação e plotagem
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=True) # Ordenar para plotagem horizontal

# Gráfico de barras da importância das features
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='darkgreen')
plt.xlabel('Importância')
plt.ylabel('Features')
plt.title('Importância de Features (Random Forests)')
plt.tight_layout()
plt.show()
# ====================================================================
# Matriz de Confusão (Item A e B) - Usando o modelo com melhor Acurácia
melhor_modelo_nome = df_resultados['Acurácia'].idxmax()
cm = resultados[melhor_modelo_nome]['Matriz_Confusao']

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Matriz de Confusão: {melhor_modelo_nome} (Item B)')
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Fracasso (0)', 'Sucesso (1)'], rotation=45)
plt.yticks(tick_marks, ['Fracasso (0)', 'Sucesso (1)'])

# Adicionar os números (anotações) no gráfico
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Previsto')
plt.tight_layout()
plt.show()