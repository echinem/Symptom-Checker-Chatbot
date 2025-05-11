from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import json
from difflib import get_close_matches
from flask_session import Session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# ====== LOAD DATA ======
nst_df = pd.read_csv("nst.csv")
nst_df['Symptoms'] = nst_df['Symptoms'].str.lower()
nst_df['Name'] = nst_df['Name'].str.lower()

label_df = pd.read_csv("textLabel.csv")
diagtext_df = pd.read_csv("diagText.csv")

with open("mapping.json", "r") as f:
    label_map = json.load(f)

# Reverse the label_map to get label -> disease name
id_to_disease = {v: k for k, v in label_map.items()}

# ====== HELPER FUNCTIONS ======
def extract_symptoms(text):
    text = text.lower()
    return [s.strip() for s in text.replace('.', '').split(',') if s.strip()]

def get_diagnosis_from_label(text):
    row = label_df[label_df['text'].str.lower() == text.lower()]
    if not row.empty:
        label = int(row.iloc[0]['label'])
        diagnosis = id_to_disease.get(label, None)
        return diagnosis
    return None

def get_treatment_for_disease(disease_name):
    if disease_name is None:
        return None
    row = nst_df[nst_df['Name'] == disease_name.lower()]
    if not row.empty:
        return row.iloc[0]['Treatments']
    return None

# ====== MAIN PREDICTION FUNCTION ======
def predict_disease(user_input):
    user_input = user_input.lower()

    # Try label-based diagnosis
    diagnosis = get_diagnosis_from_label(user_input)
    if diagnosis:
        treatment = get_treatment_for_disease(diagnosis)
        response = f"Based on your symptoms, the possible diagnosis is {diagnosis}.\n"
        if treatment:
            response += f"Recommended Treatments: {treatment}"
        else:
            response += "No treatment recommendations available."
        return response

    # Symptom-based matching
    user_symptoms = extract_symptoms(user_input)

    if 'collected_symptoms' not in session:
        session['collected_symptoms'] = []

    session['collected_symptoms'] += user_symptoms
    all_symptoms = session['collected_symptoms']

    best_match = None
    best_score = 0

    for _, row in nst_df.iterrows():
        disease_symptoms = [s.strip() for s in row['Symptoms'].split(",")]
        matched = [sym for sym in disease_symptoms if sym in all_symptoms]
        score = len(matched) / len(disease_symptoms)
        if score > best_score:
            best_score = score
            best_match = row

    if best_score >= 0.2:
        return (
            f"Your symptoms could be related to {best_match['Name']}, but this is not a diagnosis.\n"
            f"Recommended Treatments: {best_match['Treatments']}\n"
            f"Please consult a medical professional for an accurate diagnosis."
        )

    return (
        "I couldn't identify a condition based on your symptoms. "
        "Try rephrasing or listing individual symptoms like fever, cough, etc."
    )

# ====== FLASK ROUTES ======
@app.route("/")
def home():
    session.clear()
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = predict_disease(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
