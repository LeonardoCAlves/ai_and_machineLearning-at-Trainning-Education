'''
    Agora vamos carregar a base de dados e criar um modelo de regressão 
    para prever o preço das casas com base em suas características.

    Vantagem
        Simples e interpretável.
        Rápida para treinar em pequenos ou médios datasets.
        Base sólida para entender outras técnicas mais complexas.
    
    Limitações:
        Funciona melhor com relações lineares entre variáveis.
        Sensível a valores extremos (outliers).

    Resumindo, a regressão linear é um ponto de partida básico e poderoso 
    para problemas de previsão, desde que os dados sejam adequados para um modelo linear.
'''

# Importar bibliotecas necessárias
import pandas as pd  # Para manipulação e leitura dos dados
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from sklearn.linear_model import LinearRegression    # Para criar o modelo de regressão linear
from sklearn.metrics import mean_squared_error       # Para avaliar o modelo (Erro Quadrático Médio)

# Carregar a base de dados previamente salva no arquivo CSV
df = pd.read_csv(r"regressão\data\casas.csv")

# Separar as variáveis independentes (características) e dependente (preço)
X = df[["Tamanho_m2", "Quartos", "Idade"]]  # Variáveis preditoras (independentes)
y = df["Preço"]                     # Variável alvo (dependente)

# Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo com os dados de treino
model.fit(X_train, y_train)

# Fazer previsões usando o conjunto de teste
y_pred = model.predict(X_test)


'''
    Calcular o Erro Quadrático Médio (MSE) para avaliar o modelo
    O MSE calcula a diferença entre os valores reais e os valores previstos pelo modelo. 
    Depois, ele eleva essas diferenças ao quadrado (para evitar valores negativos) e tira a média. 
    O resultado nos dá uma ideia do erro médio que o modelo comete ao fazer suas previsões.

    Quando o modelo erra, o erro pode ser positivo (o modelo previu um valor maior que o real) 
    ou negativo (o modelo previu um valor menor que o real). 
    Se somarmos os erros diretamente, os valores positivos e negativos podem se cancelar, 
    dando a falsa impressão de que o modelo errou pouco. 
'''
# mse = mean_squared_error(y_test, y_pred)
# print(f"Erro Quadrático Médio (MSE): {mse:.2f}")

'''
Resultado: 
Esse valor 13.92 é a média dos erros ao quadrado entre os preços reais e os preços previstos
Como o erro está elevado ao quadrado, ele "exagera" os impactos de grandes diferenças entre 
o valor real e o previsto. 
Vamos tirar a raiz quadrada deste valor para entender melhor...
'''

# # Raiz quadrada
# print(f'Erro Médio quadrático: {round(mse ** 0.5, 2)}') 

'''
Resultado: 
Um MSE de 13.92 nos mostra que ainda há espaço para melhorar o modelo. 
Talvez incluir mais variáveis (como localização, garagem, nº de banheiros) 
ou usar um modelo mais avançado poderia reduzir esse erro.

O valor exato "bom" ou "ruim" do MSE depende do problema e das unidades dos dados. 
Se os preços das casas variam entre 1 e 20 milhões, um RMSE de 3.73 milhões representa 
um erro razoável, mas pode não ser ideal dependendo do contexto.
'''



# # Exibir os coeficientes (importância de cada variável no modelo)
# print("Coeficientes (peso das variáveis):", model.coef_)

'''
RESULTADO: 
    Lembrando que O modelo está tentando prever o preço de casas com base em três variáveis
    
    Tamanho_m2 (0.00695787):
        Para cada aumento de 1 metro quadrado no tamanho da casa, o preço da casa aumenta, 
        em média, 0.0069 milhões, ou seja, 6.957 reais 
        Isso faz sentido, já que casas maiores tendem a ser mais caras.

    Quartos (-0.17481982):
        Para cada quarto a mais na casa, o preço diminui, em média, 0.1748 milhões, 
        ou seja, 174.819 reais.
        Isso pode parecer contraintuitivo, mas indica que o número de quartos, isoladamente, 
        não está correlacionado positivamente com o preço. Talvez quartos extras estejam 
        associados a casas menores (menos área por quarto) ou a outros fatores não capturados pelo modelo.

    Idade da casa (0.04504184):
        Para cada ano a mais na idade da casa, o preço aumenta, em média, 0.045 milhões, 
        ou seja, 45.041 reais.
        Isso pode ser explicado por casas mais antigas localizadas em áreas valorizadas 
        ou com características arquitetônicas desejadas, o que eleva o preço.
'''

# # Exibir o intercepto (valor base quando todas as variáveis são zero)
# print("Intercepto (valor base):", model.intercept_)

'''
RESULTADO: 
    6.91 milhões
    Preço base da casa, é o preço da casa quando todas as Tamanho_m2, Quartos, Idade são zero.

    Como assim "zero"?
    O intercepto de 6.91 milhões indica que, se uma casa tivesse:

    0 metros quadrados
    0 quartos
    0 anos de idade
    O preço previsto para essa "casa" seria 6.91 milhões.

    O que isso significa na prática?
    Embora uma "casa" com 0 metros quadrados, 0 quartos e 0 anos de idade não exista na realidade, 
    o intercepto ainda é importante para ajustar o modelo de regressão. 
    Ele define a linha de base para as previsões.

    Exemplificando um cálculo:
    Se tivermos o seguinte para uma casa:

    Tamanho_m2 = 100
    Quartos = 3
    Idade = 10
    E a equação da regressão for:

    Preço = 6.91 + (0.00695×100) − (0.1748 × 3) + (0.045 × 10)

    O valor 6.91 (intercepto) é o preço base. 
    O modelo então adiciona ou subtrai o impacto das variáveis Tamanho_m2, Quartos e Idade 
    para estimar o preço final.

'''


# # Criar um DataFrame para comparar valores reais e previstos
results = pd.DataFrame({"Real": y_test, "Previsto": y_pred})

# Calculando a diferença entre o valor real e o valor previsto
results["Diferença"] = results["Real"] - results["Previsto"]


# # Exibir as 10 primeiras comparações
print(results.head(10))  # Exibir as primeiras linhas dos resultados

'''
RESULTADO:
    Erros grandes 
    (como em índices 44, 80 e 22) 
    indicam que o modelo pode não estar capturando bem os padrões para esses casos. 
    O modelo está errando muito nesses casos.

    Erros pequenos (como no índice 39 e 53) 
    indicam que o modelo fez boas previsões para esses exemplos, 
    com pequenas diferenças entre o valor real e o valor previsto.
'''


'''
RESUMO
    Ajuste de modelo: 
    Talvez o modelo de regressão linear não seja o mais adequado para todos os casos. 
    Outros modelos (como Random Forest ou Gradient Boosting) podem melhorar as previsões.
    
    Adicionar mais variáveis: 
    Mais informações sobre as casas, como localização, garagem, nº banheiros, condições de mercado, 
    características adicionais, etc., podem ajudar a reduzir os erros.
'''