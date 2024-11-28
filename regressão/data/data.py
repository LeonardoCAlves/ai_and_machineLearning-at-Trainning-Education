import pandas as pd
import numpy as np

# Criando um DataFrame com dados fictícios
np.random.seed(42)  # Reprodutibilidade
data = {
    "Tamanho_m2": np.random.randint(50, 300, 100),  
    "Quartos": np.random.randint(1, 5, 100),       
    "Idade": np.random.randint(1, 50, 100),         
    "Preço": np.random.uniform(0.1, 1.5, 100) * 10  
}

df = pd.DataFrame(data)

# Salvando em um arquivo CSV
df.to_csv(r"regressão\data\casas.csv", index=False)

