import joblib
import numpy as np
import pandas as pd

# Load the trained model and label encoder
rf_model_balanced = joblib.load('../models/triage_model.pkl')
label_encoder_acuity = joblib.load('../models/label_encoder_acuity.pkl')


def predict_triage(temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint):
    # Create a DataFrame with the correct feature names
    input_data = pd.DataFrame([[temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint]],
                              columns=['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain',
                                       'chiefcomplaint'])

    # Predict acuity
    acuity_encoded = rf_model_balanced.predict(input_data)
    acuity = label_encoder_acuity.inverse_transform(acuity_encoded)
    return acuity[0]


def main():
    temperature = float(input("Enter temperature: >? "))
    heartrate = float(input("Enter heartrate: >? "))
    resprate = float(input("Enter resprate: >? "))
    o2sat = float(input("Enter o2sat: >? "))
    sbp = float(input("Enter sbp: >? "))
    dbp = float(input("Enter dbp: >? "))
    pain = int(input("Enter pain (encoded as an integer): >? "))
    chiefcomplaint = int(input("Enter chiefcomplaint (encoded as an integer): >? "))

    acuity = predict_triage(temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint)
    print(f"The predicted triage level is: {acuity}")


if __name__ == "__main__":
    main()
