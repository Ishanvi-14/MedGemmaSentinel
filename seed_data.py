import json
import os

def seed_everything():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/guidelines", exist_ok=True)

    # Sample Patient
    patient_data = {
        "entry": [
            {"resource": {"resourceType": "Observation", "code": {"text": "Tumor Size"}, "valueQuantity": {"value": 25, "unit": "mm"}}},
            {"resource": {"resourceType": "Observation", "code": {"text": "CEA"}, "valueQuantity": {"value": 5.2, "unit": "ng/mL"}}}
        ]
    }
    with open("data/raw/sample_patient.json", "w") as f:
        json.dump(patient_data, f, indent=4)
    
    # Mock Guideline
    guideline_text = "NCCN NSCLC 1.2025: Progressive Disease (PD) is defined as a 20% increase in tumor size."
    with open("data/guidelines/nsclc_baseline.txt", "w") as f:
        f.write(guideline_text)
    print("✅ Data seeded in data/ folder.")

if __name__ == "__main__":
    seed_everything()