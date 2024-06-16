import joblib
import numpy as np

# Load the trained model and label encoder
rf_model_balanced = joblib.load('triage_model.pkl')
label_encoder_acuity = joblib.load('label_encoder_acuity.pkl')


def predict_triage(temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint):
    # Prepare input data
    input_data = np.array([[temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint]])

    # Predict acuity
    acuity_encoded = rf_model_balanced.predict(input_data)
    acuity = label_encoder_acuity.inverse_transform(acuity_encoded)

    return acuity[0]


# Function to get input from user and predict triage level
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
