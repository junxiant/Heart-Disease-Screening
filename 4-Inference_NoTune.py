import joblib
import numpy as np
import requests
import json


# Load the machine learning trained model from a file
def load_model(model_path="./saved_models_uci_classweights/Random Forest.joblib"):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# Load the scaler from a file
def load_preprocessors(scaler_path="./saved_models_uci_classweights/scaler.joblib"):
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully")
        return scaler
    except Exception as e:
        print(f"Error loading preprocessors: {e}")
        return None

# Mapping for categorical features
categorical_mappings = {
    'cp': ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'],
    'restecg': ['normal', 'st-t abnormality', 'lv hypertrophy'],
    'slope': ['upsloping', 'flat', 'downsloping'],
    'thal': ['normal', 'fixed defect', 'reversible defect']
}

# Prepare input data for machine learning model
def prepare_input_data(input_features):
    # Convert categorical features to numerical indices
    for feature, values in categorical_mappings.items():
        input_features[feature] = values.index(input_features[feature])

    # Define the order of features expected by the model
    feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Create a list of feature values in the correct order
    input_data = [input_features[feature] for feature in feature_order]
    input_data = np.array(input_data).reshape(1, -1)
    print("Prepared input data:", input_data)
    return input_data

# Prediction using the loaded model and preprocessor
def make_prediction(model, input_data, scaler):
    try:
        # Apply scaling to the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction and get probabilities
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)
        print(f"Raw prediction: {prediction[0]}, Probability: {probability[0]}")
        return prediction[0], probability[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

# Function to get LLM explanation using LLM
def get_llm_explanation(input_features, prediction, probability, api_url="http://172.30.240.1:1234/v1/chat/completions"):
    # Prompt for the LLM
    
    probs = f"Confidence: {probability[1]:.2f}" if prediction == 1 else probability[0]
    prompt = f"""
    A patient has the following characteristics:
    {input_features}
    
    The heart disease prediction model returned:
    Prediction: {"High risk" if prediction == 1 else "Low risk"}
    Confidence: {probs}

    Please provide:
    1. Only explain if the prediction is "High risk", else generate a "Low risk, nothing to explain".
    2. An explanation of what these features mean in the context of heart disease.
    3. Recommendations for reducing heart disease risk based on these features.
    4. Any additional insights or concerns based on this data.

    Keep the explanation short but informative.
    """

    # Prep the configs
    payload = {
        "messages": [
            {"role": "system", "content": "You are a medical AI assistant helping to interpret heart disease risk predictions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    try:
        # Start prompting
        response = requests.post(api_url, json=payload)
        response.raise_for_status() 
        
        # Extract the response
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'].strip()
        else:
            return "Error: Unable to generate explanation"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_heart_disease(input_features, model_path="./saved_models_uci_classweights/Random Forest.joblib"):
    # Load the heart disease prediction model and preprocessor
    model = load_model(model_path)
    scaler = load_preprocessors()
    if not model or not scaler:
        return

    # Prepare input data and do inference
    input_data = prepare_input_data(input_features)
    prediction, probability = make_prediction(model, input_data, scaler)

    if prediction is not None:
        # Show result
        if prediction == 1:
            print(f"The model predicts a high risk of heart disease. Confidence: {probability[1]:.2f}")
        else:
            print(f"The model predicts a low risk of heart disease. Confidence: {probability[0]:.2f}")
        
        # Get and print LLM explanation and recommendations
        llm_explanation = get_llm_explanation(input_features, prediction, probability)
        print("\nMedical LLAMA Explanation and Recommendations:")
        print(llm_explanation)
    else:
        print("Unable to make a prediction, check the input data if it is correct?")

if __name__ == "__main__":
    # Example input features (Row 1 of UCI dataset = Negative)
    input_features = {
        'age': 63,
        'sex': 1, #Male
        'cp': 'typical angina',
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 'lv hypertrophy',
        'thalch': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 'downsloping',
        'ca': 0,
        'thal': 'fixed defect'
    }
        
    # Uncomment the following block to use a positive example
    # input_features = {
    #     'age': 77,
    #     'sex': 1,  # Male
    #     'cp': 'asymptomatic',
    #     'trestbps': 300,
    #     'chol': 304,
    #     'fbs': 0,
    #     'restecg': 'lv hypertrophy',
    #     'thalch': 162,
    #     'exang': 0,
    #     'oldpeak': 0,
    #     'slope': 'upsloping',
    #     'ca': 3,
    #     'thal': 'fixed defect'
    # }
    
    # Run the prediction
    predict_heart_disease(input_features)