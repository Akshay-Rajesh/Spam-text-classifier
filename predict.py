from flask import Flask, request, jsonify, render_template
import joblib

# Define Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Serve the HTML form
@app.route("/")
def home():
    return render_template("index.html")

# Define the predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the message from the form
        message = request.form["message"]
        
        # Vectorize the message
        message_vect = vectorizer.transform([message])
        
        # Make a prediction
        prediction = model.predict(message_vect)
        
        # Map prediction to "SPAM" or "NO SPAM"
        result = "SPAM" if prediction[0] == 1 else "NO SPAM"
        
        # Return the prediction as JSON
        return jsonify({"prediction": result})
    except Exception as e:
        # Log and return error message
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
