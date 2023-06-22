from flask import Flask
import pickle

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = app.model.predict([data['input']])
    response = {'prediction': prediction[0]}
    return jsonify(response)

def create_app():
    # Load the model
    with open('rforest_regressor_price.pkl', 'rb') as file:
        app.model = pickle.load(file)
    return app