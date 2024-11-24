import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

with open("artifacts/random_forest_model.pkl", 'rb') as file:
    model = pickle.load(file)

with open("artifacts/freq_encoder.pkl", 'rb') as file:
    freq_encoder = pickle.load(file)

with open("artifacts/one_hot_encoder.pkl", 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open("artifacts/scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)


# url/
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    make = request.form.get('make')
    model_name = request.form.get('model')
    vehicle_class = request.form.get('vehicle_class')
    engine_size = float(request.form.get('engine_size'))
    cylinders = int(request.form.get('cylinders'))
    transmission = request.form.get('transmission')
    fuel = request.form.get('fuel')
    coemissions = float(request.form.get('coemissions'))
    

    df = pd.DataFrame({
        'MAKE': [make],
        'MODEL': [model_name],
        'VEHICLE CLASS': [vehicle_class],
        'ENGINE SIZE': [engine_size],
        'CYLINDERS': [cylinders],
        'TRANSMISSION': [transmission],
        'FUEL': [fuel],
        'COEMISSIONS': [coemissions]
    })

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    df_num_scaled = pd.DataFrame(
        scaler.transform(df[numerical_features]),
        columns=numerical_features,
        index=df.index
    )

    freq_cols = ['MAKE', 'MODEL']
    df_freq_encoded = pd.DataFrame(
        freq_encoder.transform(df[freq_cols]),
        index=df.index,
        columns=freq_encoder.get_feature_names_out(freq_cols)
    )

    one_hot_columns = [col for col in categorical_features if col not in freq_cols]
    df_encoded = pd.DataFrame(
    one_hot_encoder.transform(df[one_hot_columns]),
    index=df.index,
    columns=one_hot_encoder.get_feature_names_out(one_hot_columns)
)


    df = df.drop(columns=categorical_features)
    df = df.drop(columns=numerical_features)
    df_combined = pd.concat([df, df_num_scaled, df_freq_encoded, df_encoded], axis=1)

    print(df_combined.columns)
    prediction = model.predict(df_combined)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f"Predicted Fuel Consumption: {output}")


if __name__=='__main__':
    app.run(debug=True)



