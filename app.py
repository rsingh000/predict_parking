from flask import Flask, request, jsonify
import joblib

#Loading the model
with open("parking_model.pkl", "rb") as f:
    model = joblib.load(f);

app = Flask(__name__)

if __name__ == '__main__':
    app.run(port=5004)

@app.route('/')
def home():
    return "Welcome to Flask!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    hour = data['hour']
    day = data['day']
    weather = data['weather']
    events = data['events']

    prediction = model.predict([[hour,day,weather,events]])

    # Convert NumPy array to a list
    prediction_list = prediction.tolist()

    return jsonify({'prediction': prediction_list})



