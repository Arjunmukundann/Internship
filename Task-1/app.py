from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('house_price_prediction_model .pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input values from the form
            feature1 = float(request.form['feature1'])  # postcode
            feature2 = int(request.form['feature2'])  # property_type (0-4 slider)
            feature3 = int(request.form['feature3'])  # new_build (0-1 slider)
            feature4 = int(request.form['feature4'])  # freehold (0-1 slider)
            feature5 = float(request.form['feature5'])  # year
            feature6 = float(request.form['feature6'])  # month

            input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])

            # Predict
            result = model.predict(input_data)[0]
            prediction = round(result, 2)

        except Exception as e:
            prediction = f'Error: {e}'

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
