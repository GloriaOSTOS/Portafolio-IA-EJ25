import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Verificar si el archivo CSV existe
print("¿Archivo existe?", os.path.isfile("articulos_ml.csv"))

# Cargar los datos
data = pd.read_csv("articulos_ml.csv")

# Filtrar datos como antes
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]

# Crear nueva variable: suma de enlaces, comentarios e imágenes
suma = (filtered_data["# of Links"] +
        filtered_data["# of comments"].fillna(0) +
        filtered_data["# Images video"])

# Crear dataset de entrada con 2 variables
dataX2 = pd.DataFrame()
dataX2["Word count"] = filtered_data["Word count"]
dataX2["suma"] = suma

# Convertir a arrays
XY_train = np.array(dataX2)
z_train = filtered_data['# Shares'].values

# Crear modelo
regr2 = linear_model.LinearRegression()
regr2.fit(XY_train, z_train)
z_pred = regr2.predict(XY_train)

# Resultados
print("Coeficientes:", regr2.coef_)
print("Intercepto:", regr2.intercept_)
print("Error cuadrático medio: %.2f" % mean_squared_error(z_train, z_pred))
print("Puntaje de varianza (R²): %.2f" % r2_score(z_train, z_pred))

# Predicción para un artículo con 2000 palabras, 10 enlaces, 4 comentarios y 6 imágenes
entrada = [[2000, 10 + 4 + 6]]
z_dosmil = regr2.predict(entrada)
print("Predicción para artículo con 2000 palabras, 10 enlaces, 4 comentarios y 6 imágenes:", int(z_dosmil))

# Gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear malla para el plano
xx, yy = np.meshgrid(np.linspace(0, 3500, 10), np.linspace(0, 60, 10))
z = (regr2.coef_[0] * xx + regr2.coef_[1] * yy + regr2.intercept_)

# Graficar plano
ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')

# Graficar puntos reales en azul
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, c='blue', s=30)

# Graficar predicciones en rojo
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, c='red', s=40)

# Configurar vista
ax.view_init(elev=30, azim=65)
ax.set_xlabel('Cantidad de Palabras')
ax.set_ylabel('Suma: Enlaces + Comentarios + Imágenes')
ax.set_zlabel('Compartido en Redes')
ax.set_title('Regresión Lineal con Múltiples Variables')

plt.show()

