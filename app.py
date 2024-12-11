from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df1 = pd 

model = LogisticRegression()

app = Flask(__name__)
CORS(app)

def train_model():

    # Preprocesamiento del dataset
    df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')
    print("paso")
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')

    df = df.loc[df['DATE OCC']<'2021-01-01']

    # Eliminar espacios en los nombres de las columnas
    df.columns = df.columns.str.strip()

    # Intentar eliminar las columnas de nuevo
    df = df.drop(['Cross Street','Crm Cd 4', 'Crm Cd 3', 'Crm Cd 2', 'Crm Cd 1'], axis=1)
    df = df.drop(['Weapon Used Cd','Weapon Desc'], axis=1)

    df = df.dropna(subset=['Vict Sex'])
    df = df.dropna(subset=['Date Rptd'])
    df = df.dropna(subset=['Vict Descent'])
    df['Mocodes'] = df['Mocodes'].fillna('0')
    df = df.dropna(subset=['Premis Cd'])
    df['Premis Desc'] = df['Premis Desc'].fillna('NA')

    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y')

    df = df.loc[df['DATE OCC']<'2021-01-01']

    def scaleFecha(df, column):
        try:
            df[column] = pd.to_datetime(df[column],  format='%Y-%m-%d', errors='coerce')
            df[column] = df[column].dt.strftime('%Y%m%d').astype(float)
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[[column]])
            return df[column]
        except Exception as e:
            print(f"Error al procesar la columna '{column}': {e}")
    def convertirGeneroAInt(df, column):
        try:
            print(1234)
            # Crear un diccionario de mapeo único para los valores en la columna
            unique_values = df[column].dropna().unique()  # Excluir valores nulos
            mapping = {value: i for i, value in enumerate(unique_values)}

            # Reemplazar los valores en la columna con los enteros
            df[column] = df[column].map(mapping)
            return df
        except Exception as e:
            print(f"Error al procesar la columna '{column}': {e}")

    # Escalar fechas
    df['DATE OCC'] = scaleFecha(df,'DATE OCC')
    df['Date Rptd'] = scaleFecha(df, 'Date Rptd')
    df = convertirGeneroAInt(df, 'Vict Sex')
    df = convertirGeneroAInt(df, 'Vict Descent')
    global model
    global df1
    df1 = df
    df1 =  df1.drop(columns=['Date Rptd'])
    X = df[['AREA', 'Vict Descent', 'Crm Cd', 'DR_NO',  'DATE OCC', 'Part 1-2', 'Rpt Dist No' ]]  # Características
    y = df['Vict Sex']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo de regresión logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predecir usando el 20% de datos de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Crear un DataFrame con las predicciones
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # Aquí puedes añadir más pasos para entrenar el modelo con el dataframe procesado
    print("Modelo entrenado con éxito")

def on_startup():
    print("Iniciando la API...")
    train_model()

@app.route('/areas', methods=['GET'])
def get_unique_areas():
    # Eliminar duplicados basados en las columnas 'area_name' y 'area_id'
    print(12121212)
    unique_areas = df1.drop_duplicates(subset=['AREA NAME', 'AREA'])
    
    # Convertir a lista de diccionarios
    areas_list = unique_areas.to_dict(orient='records')
    
    # Retornar el listado en formato JSON
    return jsonify({'areas': areas_list})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtén los datos enviados en formato JSON
        data = request.json
        records = data['records']  # Espera una lista de listas

        # Convierte los datos a un DataFrame para procesarlos
        columns = ['AREA', 'Vict Descent', 'Crm Cd', 'DR_NO', 'DATE OCC', 'Part 1-2', 'Rpt Dist No']
        df = pd.DataFrame(records, columns=columns)
        
        def scaleFecha(df, column):
            try:
                df[column] = pd.to_datetime(df[column], errors='coerce')
                df[column] = df[column].dt.strftime('%Y%m%d').astype(float)
                scaler = MinMaxScaler()
                df[column] = scaler.fit_transform(df[[column]])
                return df
            except Exception as e:
                print(f"Error al procesar la columna '{column}': {e}")
        df = scaleFecha(df, 'DATE OCC')
        def convertirGeneroAInt(df, column):
            try:
                print(1234)
                # Crear un diccionario de mapeo único para los valores en la columna
                unique_values = df[column].dropna().unique()  # Excluir valores nulos
                mapping = {value: i for i, value in enumerate(unique_values)}

                # Reemplazar los valores en la columna con los enteros
                df[column] = df[column].map(mapping)
                return df
            except Exception as e:
                print(f"Error al procesar la columna '{column}': {e}")
        df = convertirGeneroAInt(df, 'Vict Descent')
        # Asegúrate de realizar el preprocesamiento necesario
        # Por ejemplo, transformar columnas categóricas o manejar fechas
        # df = preprocess(df)
        predictions = model.predict(df)
        results = ['mujer' if int(pred) == 0 else 'hombre' for pred in predictions]
    
        # Devuelve las predicciones en formato JSON
        return jsonify({
            'Genero Victima': results
        })
        
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    on_startup()
    app.run(debug=True)
