# Triage AI Project

# Welcome to the Triage AI Project!

## Introduction
Welcome to the Triage AI Project! This project aims to provide a machine learning solution for predicting the triage level of patients based on various medical parameters. The model helps in quickly assessing the severity of a patient's condition, ensuring timely and appropriate medical attention.

## Project Structure
Here's a quick overview of the project structure:

- `data/`: Contains the dataset files used for training and testing the models.
- `models/`: Contains the trained models and associated files.
- `scripts/`: Contains the scripts for data preprocessing, model training, and evaluation.
- `notebooks/`: Jupyter notebooks for detailed exploratory data analysis (EDA) and model development.
- `results/`: Stores the evaluation results and performance metrics.

## Getting Started

### Prerequisites
Ensure you have the following software installed:
- Python 3.7+
- Required Python packages listed in `requirements.txt`

### Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/MLer-EX/triage-ai.git
   cd triage-ai
Create a Virtual Environment:


python -m venv .venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
Install the Dependencies:


pip install -r requirements.txt
Using the Models
You can use the provided scripts to train models, evaluate them, and make predictions.

Train the Models
To train the models, use the train_model.py script:


python scripts/train_model.py
Evaluate the Models
To evaluate the models, use the evaluate_model.py script:


python scripts/evaluate_model.py
Make Predictions
To make predictions using the trained models, use the predict.py script. You can input patient data through the terminal or modify the script to accept a file input.

Example:


python scripts/predict.py
Example Command-Line Prediction Script
Here's an example script that takes patient data as input and predicts the triage level:

    import joblib
    import numpy as np

    # Load the trained model and label encoder
    rf_model_balanced = joblib.load('models/triage_model.pkl')
    label_encoder_acuity = joblib.load('models/label_encoder_acuity.pkl')

    def predict_triage(temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint):
    input_data = np.array([[temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint]])
    acuity_encoded = rf_model_balanced.predict(input_data)
    acuity = label_encoder_acuity.inverse_transform(acuity_encoded)
    return acuity[0]

    def main():
    temperature = float(input("Enter temperature: "))
    heartrate = float(input("Enter heartrate: "))
    resprate = float(input("Enter resprate: "))
    o2sat = float(input("Enter o2sat: "))
    sbp = float(input("Enter sbp: "))
    dbp = float(input("Enter dbp: "))
    pain = int(input("Enter pain (encoded as an integer): "))
    chiefcomplaint = int(input("Enter chiefcomplaint (encoded as an integer): "))
    
    acuity = predict_triage(temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint)
    print(f"The predicted triage level is: {acuity}")

    if __name__ == "__main__":
    main()

Contribution Guidelines
We welcome contributions from the community! Please fork the repository and create a pull request with your changes. Ensure that your code is well-documented and adheres to the project's coding standards.

Contact
For any questions or issues, please open an issue on GitHub or contact the project maintainer at shabanimehran@gmail.com.

Thank you for using the Triage AI Project! We hope this tool helps in providing timely and efficient medical care.


### Save the Welcome Script
Save the above markdown content into a file named `README.md` in the root directory of your project. This file serves as a comprehensive guide for anyone who wants to understand, set up, and use your Triage AI Project.

If you have any specific requests or need further customization, feel free to let me know!