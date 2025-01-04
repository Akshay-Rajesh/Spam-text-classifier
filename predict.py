import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data['message']
        message_vect = vectorizer.transform([message])
        prediction = model.predict(message_vect)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({'error':'Error processing request'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
