

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
