import pandas as pd
import pickle

def preprocess(form_data):
    form_data['color'] = form_data['color'].capitalize()
    form_data['fuelType'] = form_data['fuelType'].capitalize()
    df = pd.DataFrame([form_data.values()], columns=form_data.keys())
    pkl_file = open('color_encoder.pkl','rb')
    le_categorical = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('fuel_type_encoder.pkl','rb')
    le_fuel = pickle.load(pkl_file)
    pkl_file.close()
    df['color'] = le_categorical.transform(df['color'])
    df['fuelType'] = le_fuel.transform(df['fuelType'])
    pkl_file = open('scaler.pkl','rb')
    scaler = pickle.load(pkl_file)
    pkl_file.close()
    df_scale = scaler.transform(df)
    return df_scale