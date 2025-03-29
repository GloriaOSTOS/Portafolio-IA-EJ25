# Paso 1: Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Paso 2: Cargar el dataset
df = pd.read_csv("/Users/gloria/Desktop/reviews_sentiment.csv", sep=';')

print(df.head())

# Paso 3: Definir X y y
X = df[['wordcount','sentimentValue']].values
y = df['Star Rating'].values

# Paso 4: Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Paso 5: Escalar los datos
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Paso 6: Entrenar el modelo
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Paso 7: Evaluar precisión
print("Precisión en entrenamiento:", knn.score(X_train, y_train))
print("Precisión en prueba:", knn.score(X_test, y_test))

# Paso 8: Visualizar la segmentación de regiones
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAADD'])
cmap_bold = ['red', 'green', 'blue', 'orange', 'purple']

plt.contourf(xx, yy, Z, cmap=cmap_light)
for i in np.unique(y_train):
    plt.scatter(X_train[y_train==i][:, 0], X_train[y_train==i][:, 1], 
                label=f"{i} estrellas", color=cmap_bold[i-1], edgecolor='black')

plt.xlabel('Cantidad de Palabras (escalado)')
plt.ylabel('Sentimiento (escalado)')
plt.title('Clasificación K-NN sobre App Reviews')
plt.legend()
plt.show()

# Paso 9: Predecir una nueva muestra
nueva_muestra = scaler.transform([[20, 0.0]])  # 20 palabras, sentimiento neutral
print("Predicción:", knn.predict(nueva_muestra))
print("Probabilidades:", knn.predict_proba(nueva_muestra))
