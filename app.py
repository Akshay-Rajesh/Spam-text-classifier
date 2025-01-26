import streamlit as st
import joblib
import base64

# Custom CSS for black background and green text


# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Create a layout with two logos on the sides
col1, col2, col3 = st.columns([1, 6, 1])  # Adjust column widths

with col1:
    st.image("logo1.png", width=600)  # Replace "logo1.png" with the path to your first logo

with col3:
    st.image("logo2.png", width=600)  # Replace "logo2.png" with the path to your second logo

# App title
st.title("Spam Detection App")

# Input from the user
message = st.text_input("Enter a message to check if it's spam:")

# Predict button
if st.button("Predict"):
    try:
        # Transform the input message
        message_vect = vectorizer.transform([message])
        # Make a prediction
        prediction = model.predict(message_vect)
        # Display the result
        if prediction[0] == 1:
            st.error("This message is SPAM.")
        else:
            st.success("This message is NOT SPAM.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
