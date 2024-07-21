from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'titanic_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    data = request.form.to_dict()
    data['Age'] = float(data['Age'])
    data['SibSp'] = int(data['SibSp'])
    data['Parch'] = int(data['Parch'])
    data['Fare'] = float(data['Fare'])
    
    # Create DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Replace categorical variables with numerical equivalents
    df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
    df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)
    
    
    # Select features as per training
    final_features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
    
    # Make prediction
    prediction = model.predict(final_features)
    output = 'Survived' if prediction[0] == 1 else 'Not Survived'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
