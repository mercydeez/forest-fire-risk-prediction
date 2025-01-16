from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        temp = float(request.form['feature1'])
        oxygen = float(request.form['feature2'])
        humidity = float(request.form['feature3'])

        # Prediction and probability
        prediction = model.predict([[temp, oxygen, humidity]])
        pred_prob = model.predict_proba([[temp, oxygen, humidity]])[0][1]  # Probability of fire risk

        # Text based on prediction
        if prediction[0] == 1:
            pred_text = f"Danger: Your Forest is in Danger. Probability: {pred_prob*100:.2f}%"
        else:
            pred_text = f"Safe: Your Forest is safe for now. Probability: {(1 - pred_prob)*100:.2f}%"

        # Background context (Ensure danger is always a boolean)
        danger = bool(prediction[0] == 1)

        # Return the prediction result and danger status with probability
        return render_template('index.html', pred=pred_text, danger=danger)

    except Exception as e:
        return render_template('index.html', pred="Error in Prediction", bhai=str(e))

if __name__ == '__main__':
    app.run(debug=True)
