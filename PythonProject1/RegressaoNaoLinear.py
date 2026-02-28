import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. Carregar os dados
try:
    df = pd.read_excel("Dados.xlsx", header=1)
    df = df[['Nota_Atendimento', 'Nota_Geral']].iloc[0:10002]
    print("Dados carregados com sucesso.")
    print(df.head())
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")
    print("Verifique o nome do arquivo, o caminho e a estrutura do cabeçalho.")
    # Crie um DataFrame vazio ou saia se o carregamento falhar
    df = pd.DataFrame()


# 2. Preparar os dados
if not df.empty:
    X = df['Nota_Atendimento']
    y = df['Nota_Geral']

    # Converter para numérico e remover NaNs
    X = pd.to_numeric(X, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    data = pd.DataFrame({'X': X, 'y': y})
    data_cleaned = data.dropna()

    X_cleaned = data_cleaned['X'].values
    y_cleaned = data_cleaned['y'].values

    print(f"\nDados preparados. {len(data_cleaned)} linhas após a limpeza de NaNs.")

    # 3. Definir a função de regressão não linear exponencial
    def exponential_model(x, a, b):
      """
      Calcula o valor de y usando um modelo de regressão exponencial.

      Args:
        x: A variável independente.
        a: O coeficiente 'a' na função exponencial (y = a * exp(b * x)).
        b: O coeficiente 'b' na função exponencial (y = a * exp(b * x)).

      Returns:
        O valor calculado de y.
      """
      return a * np.exp(b * x)

    # 4. Aplicar os métodos de regressão não linear (via curve_fit)
    print("\nAplicando o método de Mínimos Quadrados Não Lineares (usando curve_fit)...")

    # Chutes iniciais para os parâmetros 'a' e 'b'
    y_positive_mask = y_cleaned > 0
    y_positive = y_cleaned[y_positive_mask]
    X_positive = X_cleaned[y_positive_mask]

    initial_guess = [1.0, 0.01] # Chutes iniciais padrão
    if len(y_positive) > 1:
      try:
        log_y_positive = np.log(y_positive)
        initial_guess_linear = np.polyfit(X_positive, log_y_positive, 1)
        initial_b = initial_guess_linear[0]
        initial_log_a = initial_guess_linear[1]
        initial_a = np.exp(initial_log_a)
        initial_guess = [initial_a, initial_b]
        print(f"Chutes iniciais estimados por linearização: a={initial_a:.4f}, b={initial_b:.4f}")
      except Exception as e:
        print(f"Erro durante a linearização para chutes iniciais: {e}")
        print("Usando chutes iniciais padrão.")
    else:
      print("Aviso: Dados positivos insuficientes para estimar chutes iniciais por linearização. Usando chutes iniciais padrão.")

    try:
      params, covariance = curve_fit(exponential_model, X_cleaned, y_cleaned, p0=initial_guess, maxfev=5000) # Increased maxfev
      optimal_a, optimal_b = params
      print(f"Parâmetros ótimos (a, b): {params}")
      # print(f"Matriz de covariância:\n{covariance}") # Opcional: exibir matriz de covariância

      # 5. Avaliar a qualidade da regressão
      print("\nAvaliando a qualidade da regressão...")
      y_pred = exponential_model(X_cleaned, optimal_a, optimal_b)
      r_squared = r2_score(y_cleaned, y_pred)
      mse = mean_squared_error(y_cleaned, y_pred)

      print(f"Métricas de Avaliação do Modelo:")
      print(f"  R-quadrado (Coeficiente de Determinação): {r_squared:.4f}")
      print(f"  Erro Quadrático Médio (MSE): {mse:.4f}")

      # 6. Visualizar a regressão
      print("\nGerando gráfico da regressão...")
      plt.figure(figsize=(10, 6))
      plt.scatter(X_cleaned, y_cleaned, label='Dados Originais', alpha=0.6)

      x_regression = np.linspace(min(X_cleaned), max(X_cleaned), 100)
      y_regression = exponential_model(x_regression, optimal_a, optimal_b)
      plt.plot(x_regression, y_regression, color='red', label=f'Curva de Regressão Exponencial\nR²={r_squared:.4f}, MSE={mse:.4f}')

      plt.xlabel('Nota de Atendimento')
      plt.ylabel('Nota Geral')
      plt.title('Regressão Exponencial: Nota Geral vs Nota de Atendimento')
      plt.legend()
      plt.grid(True)
      plt.show()

      # 7. Comparar os modelos (Explicação)
      print("\n### Comparação dos Resultados da Regressão Exponencial")
      print("\nNesta análise, realizamos uma regressão não linear exponencial para modelar a relação entre a 'Nota de Atendimento' (variável independente, X) e a 'Nota Geral' (variável dependente, y).")
      print("Utilizamos o método de Mínimos Quadrados Não Lineares, implementado pela função `curve_fit` da biblioteca SciPy.")
      print("É importante notar que `curve_fit` utiliza internamente algoritmos como Gauss-Newton e Levenberg-Marquardt para encontrar os parâmetros que melhor ajustam a curva aos dados, minimizando a soma dos quadrados dos resíduos.")

      print("\n**Parâmetros Ótimos Encontrados:**")
      print(f"- Coeficiente 'a': {optimal_a:.4f}")
      print(f"- Coeficiente 'b': {optimal_b:.4f}")
      print(f"O modelo exponencial ajustado é aproximadamente: y = {optimal_a:.4f} * exp({optimal_b:.4f} * X)")

      print("\n**Métricas de Avaliação:**")
      print(f"- R-quadrado (Coeficiente de Determinação): {r_squared:.4f}")
      print(f"- Erro Quadrático Médio (MSE): {mse:.4f}")

      print("\n**Interpretação das Métricas:**")
      print(f"O R-quadrado de {r_squared:.4f} indica a proporção da variância na variável dependente ('Nota Geral') que é previsível a partir da variável independente ('Nota de Atendimento') pelo modelo exponencial.")
      print("Um R-quadrado mais próximo de 1 sugere que o modelo explica uma grande parte da variabilidade dos dados. Um valor de 0.3031 indica que aproximadamente 30.31% da variação na Nota Geral é explicada pela Nota de Atendimento através deste modelo exponencial.")
      print(f"O Erro Quadrático Médio (MSE) de {mse:.4f} representa a média dos quadrados dos erros entre os valores reais e os valores previstos pelo modelo.")
      print("Um MSE menor indica que as previsões do modelo estão, em média, mais próximas dos valores reais. O valor de 0.1447 fornece uma medida da dispersão dos pontos em torno da curva de regressão ajustada.")

      print("\n**Visualização da Regressão:**")
      print("O gráfico gerado anteriormente mostra a dispersão dos dados originais ('Nota de Atendimento' vs 'Nota Geral') juntamente com a curva de regressão exponencial ajustada.")
      print("Esta visualização permite comparar visualmente o quão bem a curva ajustada se alinha aos dados, complementando a análise das métricas numéricas.")

      print("\n**Sobre Outros Métodos:**")
      print("A implementação direta de métodos Bayesianos puros ou de Máxima Verossimilhança pura para este tipo de regressão não linear exponencial seria significativamente mais complexa.")
      print("Esses métodos frequentemente envolvem a definição explícita da função de log-verossimilhança e, no caso Bayesiano, o uso de técnicas de amostragem como MCMC.")
      print("Portanto, para esta tarefa, focamos na aplicação do método de Mínimos Quadrados Não Lineares via `curve_fit`, que é a abordagem mais comum e acessível em bibliotecas padrão para ajuste de curvas.")

    except RuntimeError as e:
      print(f"\nErro ao ajustar a curva com curve_fit: {e}")
      print("Pode ser necessário ajustar os chutes iniciais ou verificar a adequação do modelo aos dados.")
      print("\nNão foi possível gerar as métricas de avaliação e o gráfico devido ao erro no ajuste da curva.")

else:
    print("\nNão foi possível prosseguir com a análise devido ao erro no carregamento dos dados.")