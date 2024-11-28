import pandas as pd

# Lendo arquivos
df_vendas = pd.read_excel(r'pandas\data\vendas.xlsx')
df_vendas

# Lendo arquivo de vendas corretamente e mesclando e salvando dados
df_planos = pd.read_excel(
    r'data\vendas.xlsx',
    sheet_name='Planilha2',
    skiprows=10,
    usecols=['Plano Vendido', 'Valor Mensal', 'Fk Cliente'],
    na_values=['NaN'],
    dtype={'Valor Mensal': float} 
    )

df_clientes = pd.read_excel(
    r'data\vendas.xlsx',
    sheet_name=3,
    skiprows=4,
    usecols=['ID', 'Cliente', 'Cidade'],
    na_values=['NaN']
    )

df_vendas_completo = pd.merge(
    df_clientes, df_planos, 
    left_on='ID',          
    right_on='Fk Cliente'  
)

df_vendas_completo.to_excel(
    'data/base_vendas.xlsx',
    sheet_name='Vendas',
    index=False
)


# =====================================================================


df_carros = pd.read_csv(r'pandas\data\carros.csv', delimiter=';')
df_carros


# =====================================================================


df_pedidos = pd.read_json(r'pandas\data\pedidos.json')
df_pedidos


# Analise básica
display(df_pedidos)
df_pedidos.head()
df_pedidos.head(10)
df_pedidos.columns
df_pedidos.describe
df_pedidos.describe() 
df_pedidos.info() 
df_pedidos.shape 
df_pedidos.shape[0]
df_pedidos.shape[1]
df_pedidos.dtypes
df_pedidos.sample(15) 


# faturamento
faturamento = (df_pedidos['Quantidade_Vendida'] * df_pedidos['Preco_Unitario']).sum()
faturamento


# faturamento formatado
import locale

try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8') 
    # Dólar -> en_US.UTF-8
    # Euro  -> de_DE.UTF-8

except locale.Error:
    print("Locale não suportado no sistema.")

locale.currency(faturamento, grouping=True)

