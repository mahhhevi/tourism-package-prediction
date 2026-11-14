import os
import streamlit as st
import pandas as pd
import joblib

model_path = os.path.join(
    os.path.dirname(__file__), "best_tourism_package_model_v1.joblib"
)
model = joblib.load(model_path)

st.title("Tourism Package Prediction")
st.write("Fill the customer details below to predict if they'll purchase a travel package")

Age = st.slider("Age", 18, 70, 34)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", 5, 127, 17)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 3)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 6, 4)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Number of Trips", 1, 22, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 4)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visited", 0, 3, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=22000.0, step=500.0)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

classification_threshold = 0.45

if st.button("Predict"):
    prob = float(model.predict_proba(input_data)[0, 1])
    pred = int(prob >= classification_threshold)
    if pred == 1:
        st.success(f"Prediction: Customer will purchase the travel package (Probability: {prob:.1%})")
    else:
        st.warning(f"Prediction: Customer is unlikely to purchase (Probability: {prob:.1%})")
