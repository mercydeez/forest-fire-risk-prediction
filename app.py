from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features
        temp = float(request.form['feature1'])
        oxygen = float(request.form['feature2'])
        humidity = float(request.form['feature3'])

        # Make prediction and calculate confidence
        prediction = model.predict([[temp, oxygen, humidity]])
        confidence = model.predict_proba([[temp, oxygen, humidity]])[0][1]  # Probability of fire risk

        # Round the confidence to 2 decimal places and convert to percentage
        confidence = round(confidence * 100, 2)

        # Generate the prediction message
        if prediction[0] == 1:
            pred_text = "🔥 Fire Risk: The forest is at risk of fire!"
            background_image = "/static/risk.jpeg"  # Risk background image
        else:
            pred_text = "✅ Safe: The forest is safe for now."
            background_image = "/static/no_risk.jpg"  # Safe background image

        # Pass the prediction and confidence to the template
        return render_template('index.html', pred=pred_text, confidence=confidence, background_image=background_image)

    except Exception as e:
        return render_template('index.html', pred="Error in Prediction", bhai=str(e))

if __name__ == '__main__':
    app.run(debug=True)
