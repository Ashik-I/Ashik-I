phase 4
________________________________________

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

# Split data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()






-------------------------------
phase 5
________________________________________

# healthcare_diagnostic.py

def get_diagnosis(symptoms):
    disease_database = {
        "Flu": {
            "symptoms": ["fever", "cough", "sore throat", "body ache"],
            "treatment": "Rest, hydration, and antiviral medication if severe."
        },
        "Common Cold": {
            "symptoms": ["sneezing", "stuffy nose", "sore throat", "mild cough"],
            "treatment": "Rest, fluids, and over-the-counter cold remedies."
        },
        "Malaria": {
            "symptoms": ["fever", "chills", "sweating", "headache", "nausea"],
            "treatment": "Antimalarial drugs prescribed by a physician."
        },
        "COVID-19": {
            "symptoms": ["fever", "dry cough", "fatigue", "loss of taste", "breathlessness"],
            "treatment": "Isolation, rest, fever reducers, and medical supervision if symptoms worsen."
        }
    }

    match_scores = {}
    for disease, info in disease_database.items():
        match_count = len(set(symptoms).intersection(info["symptoms"]))
        match_scores[disease] = match_count

    best_match = max(match_scores, key=match_scores.get)
    confidence = (match_scores[best_match] / len(disease_database[best_match]["symptoms"])) * 100

    return {
        "disease": best_match,
        "confidence": confidence,
        "treatment": disease_database[best_match]["treatment"]
    }

def main():
    print("Welcome to the Healthcare Diagnostic System")
    print("Enter your symptoms (comma separated): ")
    user_input = input().lower()
    symptoms = [s.strip() for s in user_input.split(',')]

    result = get_diagnosis(symptoms)

    print("\n--- Diagnosis Result ---")
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Suggested Treatment: {result['treatment']}")

if __name__ == "__main__":
    main()
