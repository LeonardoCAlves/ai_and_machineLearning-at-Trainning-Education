import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Gerando dados fictícios com faixas mais próximas
np.random.seed(42)

# Gerando mais dados (1.000 amostras)
data = {
    "Tamanho_m2": np.random.randint(50, 150, 1000),  # Tamanho em metros quadrados (faixa menor)
    "Quartos": np.random.randint(2, 4, 1000),         # Número de quartos (faixa menor)
    "Banheiros": np.random.randint(1, 3, 1000),       # Número de banheiros (faixa menor)
    "Idade": np.random.randint(1, 20, 1000),          # Idade da casa (faixa menor, casas mais novas)
    "Localizacao": np.random.choice(["Centro", "Zona Norte", "Zona Sul", "Zona Leste"], 1000),  # Localização da casa
    "Acabamento": np.random.choice(["Padrão", "Superior", "Luxo"], 1000),  # Tipo de acabamento
    "Preço": np.random.uniform(3, 7, 1000)  # Preço em milhões (faixa menor)
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Criando a coluna 'Categoria' baseada no preço
def categoria_preco(preco):
    if preco < 5:
        return 'Médio'
    else:
        return 'Alto'

df['Categoria'] = df['Preço'].apply(categoria_preco)

# Ajustando para garantir uma distribuição balanceada das classes
from sklearn.utils import resample
df_medio = df[df['Categoria'] == 'Médio']
df_alto = df[df['Categoria'] == 'Alto']

# Upsample a classe minoritária ('Alto') para balancear as classes
df_alto_upsampled = resample(df_alto, 
                              replace=True,     # Amostras com reposição
                              n_samples=len(df_medio),  # Ajusta o número de amostras
                              random_state=42)  # Reprodutibilidade

# Concatenando as classes balanceadas
df_balanced = pd.concat([df_medio, df_alto_upsampled])

# Convertendo as variáveis categóricas para variáveis numéricas (one-hot encoding)
df_balanced = pd.get_dummies(df_balanced, columns=["Localizacao", "Acabamento"], drop_first=True)

# Passo 1: Limpeza de Dados - Verificando valores ausentes
# Vamos checar se há valores ausentes nos dados
print("Valores ausentes antes da limpeza:")
print(df_balanced.isnull().sum())

# Se houver valores ausentes, vamos preencher com a média para as colunas numéricas (imputação)
imputer = SimpleImputer(strategy='mean')
df_balanced[['Tamanho_m2', 'Quartos', 'Banheiros', 'Idade', 'Preço']] = imputer.fit_transform(df_balanced[['Tamanho_m2', 'Quartos', 'Banheiros', 'Idade', 'Preço']])

# Passo 2: Normalização - Usando MinMaxScaler para normalizar as variáveis numéricas
scaler = MinMaxScaler()

# Normalizando as colunas numéricas
df_balanced[['Tamanho_m2', 'Quartos', 'Banheiros', 'Idade', 'Preço']] = scaler.fit_transform(df_balanced[['Tamanho_m2', 'Quartos', 'Banheiros', 'Idade', 'Preço']])

df_balanced.to_csv(r"classificação\data\casas.csv", index=False)


