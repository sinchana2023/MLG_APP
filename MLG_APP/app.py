from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the models and encoders
model = pickle.load(open('ensemble_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
categorical_cols = pickle.load(open('categorical_cols.pkl', 'rb'))
numerical_cols = pickle.load(open('numerical_cols.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        data = request.form.to_dict()
        
        # Print or log the received data to inspect it
        print("Data from form:", data)
        
        # Process input data
        for col in categorical_cols:
            if col in data:
                data[col] = label_encoders[col].transform([data[col]])[0]
        
        # Convert data to DataFrame
        data_df = pd.DataFrame([data])
        
        # Print or log the columns in data_df to inspect it
        print("Columns in data_df:", data_df.columns)
        
        # Remove 'Attrition' if it's mistakenly included
        if 'Attrition' in data_df.columns:
            data_df.drop('Attrition', axis=1, inplace=True)
        
        # Scale numerical columns
        data_df[numerical_cols] = scaler.transform(data_df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(data_df)
        
        # Convert prediction to human-readable format
        prediction_text = 'Employee is likely to leave.' if prediction[0] == 1 else 'Employee is likely to stay.'
        
        return render_template('result.html', prediction=prediction_text)
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
