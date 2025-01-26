import joblib
from flask import Flask, request, jsonify
import pdb

# Define FastAPI app
app = FastAPI()

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Define the input schema using Pydantic
class PredictionInput(BaseModel):
    message: str

# Define the predict endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:

        data = request.get_json()
        message = data['message']
        message_vect = vectorizer.transform([message])
        
        # Make a prediction
        prediction = model.predict(message_vect)
        
        # Return the prediction as JSON
        return {"prediction": int(prediction[0])}
    except Exception as e:
        # Log and return error message
        return HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
