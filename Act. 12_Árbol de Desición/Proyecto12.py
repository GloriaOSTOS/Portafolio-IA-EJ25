import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

# Configuración visual
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# ✅ Cargar el archivo CSV descargado
artists_billboard = pd.read_csv("artists_billboard_fix3.csv")

# Visualizaciones básicas
sb.countplot(x='artist_type', data=artists_billboard)
plt.show()

sb.countplot(x='mood', data=artists_billboard)
plt.show()

sb.countplot(x='tempo', hue='top', data=artists_billboard)
plt.show()

sb.countplot(x='genre', data=artists_billboard)
plt.show()

# Gráfico de dispersión de duración vs fecha
colores = ['orange', 'blue']
tamanios = [60, 40]

asignar = [colores[val] for val in artists_billboard['top']]
asignar2 = [tamanios[val] for val in artists_billboard['top']]

plt.scatter(artists_billboard['chart_date'], artists_billboard['durationSeg'], c=asignar, s=asignar2)
plt.axis([20030101, 20160101, 0, 600])
plt.show()

# Arreglar valores nulos en año de nacimiento
def edad_fix(anio):
    return None if anio == 0 else anio

artists_billboard['anioNacimiento'] = artists_billboard['anioNacimiento'].apply(edad_fix)

# Calcular edad del artista al momento del chart
def calcula_edad(anio, cuando):
    momento = int(str(cuando)[:4])
    return None if pd.isnull(anio) else momento - anio

artists_billboard['edad_en_billboard'] = artists_billboard.apply(
    lambda x: calcula_edad(x['anioNacimiento'], x['chart_date']), axis=1
)

# Rellenar valores nulos en edad con datos aleatorios
age_avg = artists_billboard['edad_en_billboard'].mean()
age_std = artists_billboard['edad_en_billboard'].std()
age_null_count = artists_billboard['edad_en_billboard'].isnull().sum()

age_null_random_list = np.random.randint(
    age_avg - age_std, age_avg + age_std, size=age_null_count
)

conValoresNulos = np.isnan(artists_billboard['edad_en_billboard'])
artists_billboard.loc[conValoresNulos, 'edad_en_billboard'] = age_null_random_list
artists_billboard['edad_en_billboard'] = artists_billboard['edad_en_billboard'].astype(int)

# Mostrar estadísticos
print("Edad Promedio:", round(age_avg, 2))
print("Desvío Estándar Edad:", round(age_std, 2))

# Visualización de edad con colores por 'top'
colores = ['orange', 'blue', 'green']
asignar = [
    colores[2] if conValoresNulos[i] else colores[row]
    for i, row in enumerate(artists_billboard['top'])
]

plt.scatter(artists_billboard['edad_en_billboard'], artists_billboard.index, c=asignar, s=30)
plt.axis([15, 50, 0, 650])
plt.show()

# ✅ Árbol de Decisión
# Seleccionar características para entrenamiento
X = artists_billboard[['durationSeg', 'anioNacimiento', 'edad_en_billboard']]
Y = artists_billboard['top']

# Validación cruzada
kf = KFold(n_splits=10, shuffle=True, random_state=1)
modelo = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)

# Evaluar con cross_val_score
scores = cross_val_score(modelo, X, Y, cv=kf)
print("Precisión promedio (cross-validation):", round(scores.mean(), 4))

# Entrenar modelo completo y graficar árbol
modelo.fit(X, Y)
tree.plot_tree(modelo, filled=True, feature_names=X.columns, class_names=["No Top", "Top"])
plt.show()
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

# Configuración visual
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# ✅ Cargar el archivo CSV descargado
artists_billboard = pd.read_csv("artists_billboard_fix3.csv")

# Visualizaciones básicas
sb.countplot(x='artist_type', data=artists_billboard)
plt.show()

sb.countplot(x='mood', data=artists_billboard)
plt.show()

sb.countplot(x='tempo', hue='top', data=artists_billboard)
plt.show()

sb.countplot(x='genre', data=artists_billboard)
plt.show()

# Gráfico de dispersión de duración vs fecha
colores = ['orange', 'blue']
tamanios = [60, 40]

asignar = [colores[val] for val in artists_billboard['top']]
asignar2 = [tamanios[val] for val in artists_billboard['top']]

plt.scatter(artists_billboard['chart_date'], artists_billboard['durationSeg'], c=asignar, s=asignar2)
plt.axis([20030101, 20160101, 0, 600])
plt.show()

# Arreglar valores nulos en año de nacimiento
def edad_fix(anio):
    return None if anio == 0 else anio

artists_billboard['anioNacimiento'] = artists_billboard['anioNacimiento'].apply(edad_fix)

# Calcular edad del artista al momento del chart
def calcula_edad(anio, cuando):
    momento = int(str(cuando)[:4])
    return None if pd.isnull(anio) else momento - anio

artists_billboard['edad_en_billboard'] = artists_billboard.apply(
    lambda x: calcula_edad(x['anioNacimiento'], x['chart_date']), axis=1
)

# Rellenar valores nulos en edad con datos aleatorios
age_avg = artists_billboard['edad_en_billboard'].mean()
age_std = artists_billboard['edad_en_billboard'].std()
age_null_count = artists_billboard['edad_en_billboard'].isnull().sum()

age_null_random_list = np.random.randint(
    age_avg - age_std, age_avg + age_std, size=age_null_count
)

conValoresNulos = np.isnan(artists_billboard['edad_en_billboard'])
artists_billboard.loc[conValoresNulos, 'edad_en_billboard'] = age_null_random_list
artists_billboard['edad_en_billboard'] = artists_billboard['edad_en_billboard'].astype(int)

# Mostrar estadísticos
print("Edad Promedio:", round(age_avg, 2))
print("Desvío Estándar Edad:", round(age_std, 2))

# Visualización de edad con colores por 'top'
colores = ['orange', 'blue', 'green']
asignar = [
    colores[2] if conValoresNulos[i] else colores[row]
    for i, row in enumerate(artists_billboard['top'])
]

plt.scatter(artists_billboard['edad_en_billboard'], artists_billboard.index, c=asignar, s=30)
plt.axis([15, 50, 0, 650])
plt.show()

# ✅ Árbol de Decisión
# Seleccionar características para entrenamiento
X = artists_billboard[['durationSeg', 'anioNacimiento', 'edad_en_billboard']]
Y = artists_billboard['top']

# Validación cruzada
kf = KFold(n_splits=10, shuffle=True, random_state=1)
modelo = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)

# Evaluar con cross_val_score
scores = cross_val_score(modelo, X, Y, cv=kf)
print("Precisión promedio (cross-validation):", round(scores.mean(), 4))

# Entrenar modelo completo y graficar árbol
modelo.fit(X, Y)
tree.plot_tree(modelo, filled=True, feature_names=X.columns, class_names=["No Top", "Top"])
plt.show()


