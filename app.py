from flask import Flask, render_template, jsonify, request
import pickle
import pandas as pd
from preprocessor import preprocess

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('car_specifications_input.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    form_data = dict(request.form)
    preprocessed_data = preprocess(form_data)
    lr = pickle.load(open('rforest_regressor_price.pkl',"rb"))
    prediction = lr.predict(preprocessed_data)
    return render_template("prediction_page.html",prediction=prediction[0])

if __name__=="__main__":
    app.run(debug=True)