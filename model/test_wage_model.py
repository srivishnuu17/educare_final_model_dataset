
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("education_wage_model.pkl")

# Load your original dataset
df = pd.read_csv("slm_dataset.csv.csv")

# Features used during training
features = [
    "cip_title",
    "credential_name",
    "cip_family_title",
    "occupation_typical_entry_education",
    "occupation_employment_2024_thousands",
    "occupation_growth_pct_2024_34",
    "occupation_annual_openings_thousands",
    "linked_occupations_count",
    "ai_software_occupation_share",
    "expert_system_occupation_share",
    "hot_technologies_mean_per_occupation"
]

# Select a record with all required input features
test_data = df.dropna(subset=features).iloc[[0]]

# Predict annual occupation wage
prediction = model.predict(test_data[features])[0]

print("\n--- WAGE PREDICTION TEST ---")
print("Education Program:", test_data["cip_title"].iloc[0])
print("Credential:", test_data["credential_name"].iloc[0])
print("Predicted Annual Occupation Wage: $", round(float(prediction), 2))
print("Model test completed successfully!")
