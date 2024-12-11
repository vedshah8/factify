from flask import Flask, request, render_template
import pickle
from scipy.sparse import hstack

# Load the model and vectorizer
with open('text_vectorizer.pkl', 'rb') as file:
    text_vectorizer = pickle.load(file)

with open('trained_model.pkl', 'rb') as file:
    trained_model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    title = request.form['title']
    text = request.form['text']

    # Vectorize inputs
    text_vectorized = text_vectorizer.transform([text])
    title_vectorized = text_vectorizer.transform([title])

    # Combine vectors
    weighted_input = hstack([text_vectorized * 1.0, title_vectorized * 1.0])

    # Make prediction
    prediction = trained_model.predict(weighted_input)[0]
    label = 'Real News' if prediction == 1 else 'Fake News'

    return render_template('index.html', prediction_text=f'The news is: {label}')

if __name__ == "__main__":
    app.run(debug=True)
