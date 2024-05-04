import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('lr_model.joblib')

# Define route for the home page
@app.route('/')
def home():
    """
    Renders the home page.
    """
    return render_template('first.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST request for prediction.
    """
    if request.method == 'POST':
        # Get the input data from the form
        features = [float(x) for x in request.form.values()]
        
        # Convert the input data into a numpy array
        input_features = np.array(features).reshape(1, -1)
        
        # Make prediction using the pre-trained model
        prediction = model.predict(input_features)
        
        # Pass prediction value to the template
        return render_template('first.html', prediction=int(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
