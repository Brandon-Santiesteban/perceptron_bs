from email import header
from operator import index
import pandas as pd
import numpy as np

# absolute_path = "C:\\Users\\brasa\\OneDrive\\Universidad\\2022 - 1\\Sistemas Inteligentes - Chachon\\Python Chacon\\perceptron_data.xlsx"
relative_path = "perceptron_data.xlsx"

# leer xlsx
# xls = pd.ExcelFile(absolute_path)
xls = pd.ExcelFile(relative_path)
    
df = xls.parse('database')

a = np.array(df)

# procesamiento de datos
def funcion_activacion_paso(w, x, b):
    z = w * x
    if z.sum() + b > 0:
        return 1
    else:
        return 0


# entrenamiento perceptron
def epocas_():
    pesos = np.random.uniform(0, 1, size=2)
    tasa_aprendizaje = 0.01
    # umbral
    b = np.random.uniform(0, 1)
    # cant iteacion de entrenamiento
    epocas = 100
    # ciclo de entrenemiento
    for epoca in range(epocas):
        error_total = 0
        # iteracione de array
        for i in range(len(a)):
            prediccion = funcion_activacion_paso(pesos, a[i][0], b)
            error = a[i][2] - prediccion
            error_total += error**2
            pesos[0] += tasa_aprendizaje * a[i][0] * error
            pesos[1] += tasa_aprendizaje * a[i][1] * error
            b += tasa_aprendizaje * error
        print(int(error_total), end=' ')
    print('\nResultado: ' + str(funcion_activacion_paso(pesos, [0.2, 0.9], b)))


if __name__ == '__main__':
    epocas_()


# guardar archivo excel
df.to_excel('perceptron_data.xlsx', index = False, header = False)

# ahora agregar al data frame el nuevo dato con el que se evaluo como dato de entrenamiento
# esto con el fin de hacer mas integeligente el perceptron entre mas datos se evaluen y se agg a la bd
