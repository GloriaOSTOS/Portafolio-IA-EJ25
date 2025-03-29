# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Cargar dataset
df = pd.read_csv('/Users/gloria/Desktop/creditcard.csv')


# Ver balance de clases
print(df['Class'].value_counts())

# Separar variables predictoras y objetivo
X = df.drop('Class', axis=1)
y = df['Class']

# Escalar las variables (muy recomendable en este dataset)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Modelo 1: Random Forest
model_rf = RandomForestClassifier(n_estimators=100,
                                  bootstrap=True,
                                  max_features='sqrt',
                                  random_state=42,
                                  verbose=1)
model_rf.fit(X_train, y_train)

# Predicciones
y_pred_rf = model_rf.predict(X_test)

# Evaluación
print("Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Modelo 2: Baseline con Regresión Logística (para comparar)
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("Regresión Logística (Baseline):")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Comparación visual (opcional)
from sklearn.metrics import roc_curve, auc

# Calcular probabilidades
probs_rf = model_rf.predict_proba(X_test)[:, 1]
probs_lr = model_lr.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, probs_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)

auc_rf = auc(fpr_rf, tpr_rf)
auc_lr = auc(fpr_lr, tpr_lr)

plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.legend()
plt.grid()
plt.show()
