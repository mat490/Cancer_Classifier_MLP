import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

df_cancer_cells = pd.read_csv('/Cancer_Data.csv')

# Convertimos la columna "diagnosis" de df_cancer_cells en valores numericos cambiando las "M" por 1 y las "B" por 0
df_cancer_cells['diagnosis'].replace(['M','B'],[1,0],inplace=True)

# Separamos nuestras entradas y nuestras salidas
x_df_cc = df_cancer_cells.iloc[:,2:]
y_df_cc = df_cancer_cells['diagnosis']

# En nuestra muestra separamos nuestros datos de entrenamiento y nuestros datos de salida
X_train, X_test, y_train, y_test = train_test_split(x_df_cc, y_df_cc, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential()
model.add(Dense(units= 12, activation='relu', input_dim=30))  # Ajusta input_dim al número de características
model.add(Dense(units=1, activation='sigmoid'))

# Compilamos el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamos el modelo
entrenar_modelo()

def entrenar_modelo():
  model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
  
  # Verificamos la presición
  test_loss, test_accuracy = model.evaluate(X_test, y_test)
  print(f'\nTest Accuracy: {test_accuracy*100:.2f}%')
  
  # Guardamos el modelo si la presición es mayor o igual a 0.95
  if (test_accuracy >= 0.95):
  
    model.save('modelo_preentrenado_cc.h5')
  else:
    entrenar_modelo()