# Importndo Numpy
import numpy as np

# Criando um array 1D
arr1 = np.array([1, 3, 5, 7, 9])
arr2 = np.array([2, 4, 6, 8, 10])
arr3 = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])

arr4 = np.array([
    [2, 2, 2, 2, 2, 2, 2],
    [4, 4, 4, 4, 4, 4, 4],
    [8, 8, 8, 8, 8, 8, 8]
])


# Atributos de um array
arr3.shape # Retorna as dimensões do array.
arr3.ndim  # Número de dimensões.1
arr3.dtype # Tipo dos elementos.
arr3.size  # Número total de elementos.


# Slice (Fatiamento)
arr3[1, 2] # linha 1, coluna 2 Lembrando que começa do 0 internamente


# Operações aritméticas
soma = arr1 + arr2
soma

soma = list(arr1) + list(arr2)
soma


# Multiplicando valores sem uso de loops
arr3 * 2 


# Calculando raiz quadrada de cada elemento
np.sqrt(np.array([1, 4, 9, 16]))


# Criando arrays apartir de funções
np.zeros()    # Cria um array de zeros.
np.ones()     #  Cria um array de uns.
np.arange()   # Cria um array com valores espaçados de acordo com o intervalo fornecido.
np.linspace() #  Cria um array com números distribuídos uniformemente entre um intervalo.

zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
range_array = np.arange(0, 10, 2)
linspace_array = np.linspace(0, 1, 5)

print(zeros)
print(ones)
print(range_array)
print(linspace_array)


# Multiplicação de Matrizes (Produto Escalar)
arr10 = np.array([[1, 2], [3, 4]])
arr20= np.array([[5, 6], [7, 8]])
produto = np.dot(arr10, arr20)
print(produto)



