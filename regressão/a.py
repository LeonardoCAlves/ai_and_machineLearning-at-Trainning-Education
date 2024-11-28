from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Carregar a base de dados expandida
df = pd.read_csv(r"regressão\data\casas.csv")

# Variáveis independentes (incluímos as novas variáveis)
X = df[["Tamanho_m2", "Quartos", "Idade", "Banheiros", "Garagem", "Localização", "AnoRenovacao"]]

# Variável dependente (preço)
y = df["Preço"]

# Dividir em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Raiz quadrada do MSE
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Erro Médio Quadrático (RMSE): {rmse:.2f}")

# Exibir os coeficientes do modelo
coef = pd.DataFrame({"Variável": X.columns, "Coeficiente": model.coef_})
print(coef)

