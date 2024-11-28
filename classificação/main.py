import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregando o arquivo de dados
df = pd.read_csv(r"classificação\data\casas.csv")  # Lê o conjunto de dados salvo anteriormente, que contém informações das casas e suas categorias.

# Separando as variáveis independentes (X) e a variável dependente (y)
X = df.drop(columns=["Preço", "Categoria"])  # Exclui as colunas 'Preço' (não relevante para a classificação) e 'Categoria' (variável alvo).
y = df["Categoria"]  # Define a variável dependente (alvo) como a coluna 'Categoria', que contém as classes para classificação.

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                # Divide os dados em variáveis independentes e dependente
    test_size=0.3,       # 30% dos dados serão usados como conjunto de teste, 70% para treinamento.
    random_state=42      # Define um valor para garantir reprodutibilidade na divisão.
)

'''
random_state:
    Quando trabalhamos com aprendizado de máquina, muitas vezes lidamos com processos 
    que envolvem aleatoriedade. Por exemplo:

    Quando dividimos nossos dados em treino e teste, os registros são selecionados de forma aleatória.
    Modelos como a Random Forest criam várias árvores de decisão usando amostras aleatórias do conjunto 
    de treinamento.

    Agora, imaginem que a cada vez que rodamos o código, o computador escolhe os dados ou as amostras 
    de forma diferente. Isso pode ser um problema porque os resultados vão mudar toda vez que rodarmos 
    o modelo. Fica difícil comparar resultados, corrigir problemas ou explicar o que aconteceu.

    É aí que entra o random_state. Ele funciona como uma 'semente' que controla a aleatoriedade. 
    'Mesmo que seja aleatório, faça sempre do mesmo jeito.' 

    "Vocês devem ter reparado que o valor usado no código foi 42. Sabem por quê? 
    É uma brincadeira com o livro O Guia do Mochileiro das Galáxias, que diz que 42 é a resposta 
    para a pergunta fundamental da vida, do universo e tudo mais.

    No livro, um supercomputador chamado Pensador Profundo (Deep Thought) é construído para responder 
    à "Pergunta Fundamental da Vida, do Universo e de Tudo". 
    
    Após 7,5 milhões de anos de cálculos, ele finalmente revela que a resposta é... 42.

    Só que tem um problema: ninguém sabe qual é exatamente a Pergunta Fundamental. 
    Eles pediram a resposta sem entender a pergunta! 😂

    Para resolver isso, um outro computador gigante é construído: a Terra. 
    Segundo o livro, nosso planeta era, na verdade, um experimento para descobrir essa pergunta, 
    mas foi destruído pelos vogons antes que pudéssemos obter o resultado.

'''

'''
    Normalizando os dados numéricos
        Se não normalizarmos, essas diferenças de escala podem influenciar o modelo, 
        fazendo com que ele dê mais importância às variáveis com valores maiores, 
        mesmo que elas não sejam mais relevantes.    
'''
# Inicializa o escalador padrão, que normaliza os dados para ter média 0 e desvio padrão 1.
scaler = StandardScaler()  

'''
    Imagine que você está transformando os dados para uma "escala uniforme", 
    como se estivesse ajustando os ponteiros de um velocímetro, para que ele funcione 
    corretamente independente do carro. 
    
    Essa padronização ajuda os algoritmos a interpretar 
    as variáveis de forma justa, sem dar prioridade indevida a alguma delas.
'''

# Ajusta e transforma os dados de treinamento, e apenas transforma os dados de teste
# Aplica normalização apenas às colunas numéricas.
X_train[['Tamanho_m2', 'Quartos', 'Banheiros', 'Idade']] = scaler.fit_transform(
    X_train[['Tamanho_m2', 'Quartos', 'Banheiros', 'Idade']]  
)

# Normaliza as mesmas colunas nos dados de teste usando os parâmetros ajustados no treino.
X_test[['Tamanho_m2', 'Quartos', 'Banheiros', 'Idade']] = scaler.transform(
    X_test[['Tamanho_m2', 'Quartos', 'Banheiros', 'Idade']]  
)

# Inicializando e treinando o modelo de Random Forest

'''
Árvores de decisão:
    Cada árvore de decisão é um modelo que divide os dados em grupos menores com base 
    em perguntas feitas em sequência. Por exemplo: "O tamanho da casa é maior que 100m²?" 
    ou "O número de quartos é maior que 3?"

    Essas perguntas criam ramificações, terminando em previsões chamadas folhas.

Vários modelos, um só resultado:
    O Random Forest cria várias árvores de decisão (daí o "floresta"). 
    Cada árvore é treinada em uma amostra aleatória dos dados e usa atributos diferentes. 
    Isso introduz variação, reduzindo a chance de "overfitting" 
    (quando o modelo fica bom só nos dados de treino, mas ruim em novos dados).

Combinação dos resultados:
    Após treinar todas as árvores, o Random Forest faz uma votação: cada árvore "vota" 
    na classe que acha correta, e a classe com mais votos é a previsão final.
'''

classifier = RandomForestClassifier(
    n_estimators=100,    # Usa 100 árvores de decisão na floresta aleatória.
    random_state=42      # Define um valor fixo para garantir que os resultados sejam reprodutíveis.
)
classifier.fit(X_train, y_train)  # Treina o modelo nos dados de treinamento (X_train, y_train).

'''
    Imagine que cada árvore de decisão é como um "consultor especialista" 
    em tomar decisões com base nos dados. 
    Se você consulta apenas um especialista, ele pode errar. 
    Mas ao consultar 100 especialistas diferentes, suas chances de tomar a 
    decisão correta aumentam, porque o erro de um será compensado pelos outros. 
    
    Essa é a ideia por trás do Random Forest!

    "Um júri decidindo um caso, onde cada jurado é uma árvore de decisão."
'''


# Fazendo previsões no conjunto de teste
y_pred = classifier.predict(X_test)  # Faz previsões com base nos dados de teste (X_test).

# Avaliando o modelo
# Calcula a acurácia comparando os valores reais (y_test) e previstos (y_pred).
accuracy = accuracy_score(y_test, y_pred)  
print(f"Acurácia do modelo de classificação: {accuracy * 100:.2f}%")  # Imprime a acurácia do modelo formatada como porcentagem.
