from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load your model (Ensure model.pkl is in the same directory as app.py)
# If you have a scaler.pkl, load that as well to transform inputs
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl not found. Please ensure your trained model is in the project folder.")

@app.route('/')
def home():
    """Renders the main landing page with the form and project info."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission, processes inputs, and returns the 
    prediction result to the same page.
    """
    try:
        # 1. Capture inputs from the enhanced HTML form
        # We use float() to ensure numerical data for the ML model
        amount = float(request.form['amount'])
        history = float(request.form['history'])
        ins_type = float(request.form['type'])
        
        # 2. Prepare feature array
        # Based on your previous errors, your model expects 39 features.
        # We initialize a zero-vector and map your 3 inputs to it.
        # If your training used a different order, adjust index 0, 1, 2 accordingly.
        features = np.zeros(39)
        features[0] = amount
        features[1] = history
        features[2] = ins_type
        
        # 3. Perform Prediction
        # model.predict expects a 2D array: [ [f1, f2, ... f39] ]
        prediction = model.predict(features.reshape(1, -1))
        
        # 4. Format the result string
        # This string is checked by the CSS logic to decide color (Red/Green)
        if prediction[0] == 1:
            result = "Fraudulent Claim Detected"
        else:
            result = "Genuine Claim"
            
        # 5. Return to index.html with the prediction_text variable
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        # Catch errors (like missing inputs) and display them safely in the UI
        return render_template('index.html', prediction_text=f"Execution Error: {str(e)}")

if __name__ == "__main__":
    # debug=True allows the server to auto-reload when you save changes
    app.run(debug=True)