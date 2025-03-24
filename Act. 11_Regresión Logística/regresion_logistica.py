import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
import os

# Verificar si el archivo existe
archivo = "usuarios_win_mac_lin.csv"
print("¿Archivo encontrado?", os.path.exists(archivo))

# Cargar el dataset
dataframe = pd.read_csv(archivo)
print(dataframe.head())
print(dataframe.describe())

# Ver distribución de clases
print(dataframe.groupby('clase').size())

# Visualización de histogramas
dataframe.drop(['clase'], axis=1).hist()
plt.show()

# Visualización de pares de variables
sb.pairplot(dataframe.dropna(), hue='clase', vars=["duracion", "paginas", "acciones", "valor"])
plt.show()

# Preparar datos
X = np.array(dataframe.drop(['clase'], axis=1))
y = np.array(dataframe['clase'])
print("Dimensión de X:", X.shape)

# Crear y entrenar el modelo
model = linear_model.LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predicciones y evaluación
predictions = model.predict(X)
print("Primeras predicciones:", predictions[:5])
print("Precisión sobre todo el conjunto:", model.score(X, y))

# Separar datos para validación
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, y, test_size=validation_size, random_state=seed)

# Validación cruzada
name = 'Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
cv_result = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_result.mean(), cv_result.std())
print(msg)

# Validar con subconjunto
predictions = model.predict(X_validation)
print("Precisión en validación:", accuracy_score(Y_validation, predictions))
print("Matriz de confusión:\n", confusion_matrix(Y_validation, predictions))
print("Reporte de clasificación:\n", classification_report(Y_validation, predictions))

# Clasificar un nuevo usuario
X_new = pd.DataFrame({'duracion': [10], 'paginas': [3], 'acciones': [5], 'valor': [9]})
prediccion_nueva = model.predict(X_new)
print("Predicción para nuevo usuario:", prediccion_nueva)  # Resultado: 0, 1 o 2

