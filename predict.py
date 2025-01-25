import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict

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
        # Extract message from the input data
        message = input_data.message
        
        # Vectorize the message
        message_vect = vectorizer.transform([message])
        
        # Make a prediction
        prediction = model.predict(message_vect)
        
        # Return the prediction as JSON
        return {"prediction": int(prediction[0])}
    except Exception as e:
        # Log and return error message
        return HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Example root endpoint to test the server
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Prediction API!"}

# Run the app (if running directly, e.g., for local testing)
# Uncomment the lines below if using `python predict.py` to start
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=80)
