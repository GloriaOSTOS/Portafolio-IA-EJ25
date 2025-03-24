import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt  # ← corregido: era 'matplotlib.pyplotas'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score  # ← añadiste mean_absolute_error, pero usas MSE

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Cargamos los datos de entrada
data = pd.read_csv("articulos_ml.csv")

# Mostramos forma y primeras filas
print("Dimensiones:", data.shape)
print(data.head())

# Estadísticas
print(data.describe())

# Visualizamos características
data.drop(['Title', 'url', 'Elapsed days'], axis=1).hist()
plt.show()

# Filtramos los datos
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]

colores = ['orange', 'blue']
tamanios = [30, 60]
f1 = filtered_data['Word count'].values
f2 = filtered_data['# Shares'].values

# Pintamos por encima y debajo de la media
asignar = []
for index, row in filtered_data.iterrows():
    if row['Word count'] > 1808:
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])

plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.xlabel("Word Count")
plt.ylabel("# Shares")
plt.title("Dispersión de Palabras vs Compartidos")
plt.show()

# Regresión lineal simple
dataX = filtered_data[["Word count"]]
X_train = np.array(dataX)
y_train = filtered_data['# Shares'].values

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_train)

# Resultados
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
print("Variance score (R²): %.2f" % r2_score(y_train, y_pred))

# Predicción para artículo de 2000 palabras
y_dosmil = regr.predict([[2000]])
print("Predicción para artículo de 2000 palabras:", int(y_dosmil))
