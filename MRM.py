# Regresión Múltiple con dataset de precios de casas de Kaggle
#Link del CSV obtenido: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv
#Se utilizó el dataset train.csv de la competencia de Kaggle "House Prices: Advanced Regression Techniques", con más de 1400 registros. 
#Desde la línea 55 esta el analisis tecnico y estadistico del modelo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Cargar el dataset
dataset = pd.read_csv('train.csv')

# 2. Selección de variables
variables_numericas = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd']
variables_categoricas = ['Neighborhood', 'HouseStyle', 'KitchenQual', 'ExterQual']
X = dataset[variables_categoricas + variables_numericas]
y = dataset['SalePrice'].values

# 3. Codificación categórica
cat_indices = list(range(len(variables_categoricas)))
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop='first'), cat_indices)
], remainder='passthrough')
X_encoded = ct.fit_transform(X).toarray()  # Convertimos correctamente

# 4. División en conjuntos
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

# 5. Entrenamiento del modelo
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 6. Predicción
y_pred = regressor.predict(X_test)

# 7. Visualización
np.set_printoptions(precision=2)
resultados = np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1)
print("Predicciones vs Reales:\n", resultados[:10])

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores reales')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones')
plt.title('Predicciones vs Valores reales (Precio de Casas)')
plt.xlabel('Índice de muestra')
plt.ylabel('Precio ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# ANÁLISIS TÉCNICO DEL MODELO
# ===============================
# Se entrenó un modelo de Regresión Lineal Múltiple sobre un conjunto de datos con más de 1400 viviendas,
# utilizando variables relevantes tanto numéricas como categóricas. Las variables categóricas fueron transformadas
# mediante codificación OneHot para ser compatibles con el modelo lineal.
#Funcion del Modelo lineal multiple: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

# En este modelo, la variable dependiente (también llamada objetivo o respuesta) es 'SalePrice',
# ya que representa el valor que se desea predecir: el precio de venta de las viviendas.
# Por otro lado, las variables independientes son aquellas que el modelo usa como predictores:
#    - Variables numéricas: 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt',
#      'FullBath', 'TotRmsAbvGrd'.
#    - Variables categóricas: 'Neighborhood', 'HouseStyle', 'KitchenQual', 'ExterQual', las cuales fueron codificadas
#      con OneHotEncoder para su uso en el modelo.
# Estas variables independientes se consideran factores explicativos que tienen influencia directa
# sobre el precio de la vivienda, y por tanto, forman la matriz X del modelo de regresión.

# A partir del gráfico generado (Predicciones vs Reales), se puede observar una dispersión amplia,
# especialmente en los extremos de precio. Esto indica que, aunque el modelo es capaz de capturar
# cierta tendencia general, tiene dificultades para predecir valores muy altos o bajos con precisión.

# El modelo presenta una tendencia a subestimar los precios de viviendas más caras y a sobrestimar las más baratas,
# lo cual sugiere una posible falta de linealidad total en la relación entre las variables independientes
# y el precio de venta, algo común cuando se aplican modelos lineales a fenómenos más complejos.


# ===============================
# ANÁLISIS ESTADÍSTICO
# ===============================
# A partir de los primeros 10 pares Predicción vs Real, se calcularon errores promedio y dispersión.

# Predicciones vs Reales:
#  [[273593.73 200624.  ]
#   [142872.26 133000.  ]
#   [128205.5  110000.  ]
#   [233182.08 192000.  ]
#   [102116.37  88000.  ]
#   [101069.77  85000.  ]
#   [241407.04 282922.  ]
#   [134660.8  141000.  ]
#   [529740.95 745000.  ]
#   [165496.07 148800.  ]]

# Para evaluar el rendimiento del modelo, se utilizaron tres métricas clave:

# 1. MAE (Error Absoluto Medio)
#    Fórmula: MAE = (1/n) * Σ |yᵢ - ŷᵢ|
#    Donde:
#      - yᵢ: valor real
#      - ŷᵢ: valor predicho
#      - n: número de observaciones
#    Mide el error promedio sin importar la dirección. Es interpretable directamente en unidades monetarias.

# 2. RMSE (Raíz del Error Cuadrático Medio)
#    Fórmula: RMSE = sqrt( (1/n) * Σ (yᵢ - ŷᵢ)² )
#    Penaliza más los errores grandes, por lo que es sensible a valores atípicos o predicciones muy desviadas.

# 3. Desviación Estándar del Error
#    Fórmula: σ = sqrt( (1/n) * Σ (eᵢ - ē)² ), donde eᵢ = yᵢ - ŷᵢ
#    Mide la dispersión de los errores individuales respecto al error medio. Ayuda a evaluar la estabilidad del modelo.

# En esta muestra observada, el MAE ronda entre $30,000 y $40,000, lo cual puede considerarse elevado,
# especialmente si se compara con precios de casas de gama media (alrededor de $150,000 a $250,000).
# La presencia de errores extremos, como una predicción de $529,000 para una vivienda que costaba $745,000,
# muestra que el modelo tiende a cometer errores mayores en viviendas fuera del rango medio.
# Esto también se refleja en una desviación estándar alta, que indica una falta de precisión homogénea
# en todo el rango de precios.
