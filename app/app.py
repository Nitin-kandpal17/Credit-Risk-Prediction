from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model/final_xgb_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            input_data = {
                "person_age": float(request.form["person_age"]),
                "person_income": float(request.form["person_income"]),
                "person_emp_length": float(request.form["person_emp_length"]),
                "loan_amnt": float(request.form["loan_amnt"]),
                "loan_int_rate": float(request.form["loan_int_rate"]),
                "loan_percent_income": float(request.form["loan_percent_income"]),
                "cb_person_cred_hist_length": float(request.form["cb_person_cred_hist_length"]),
                "debt_to_income_ratio": float(request.form["loan_amnt"]) / float(request.form["person_income"]),
                "person_home_ownership_RENT": 1 if request.form["person_home_ownership"] == "RENT" else 0,
                "person_home_ownership_OWN": 1 if request.form["person_home_ownership"] == "OWN" else 0,
                "person_home_ownership_OTHER": 1 if request.form["person_home_ownership"] == "OTHER" else 0,
                "loan_intent_EDUCATION": 1 if request.form["loan_intent"] == "EDUCATION" else 0,
                "loan_intent_HOMEIMPROVEMENT": 1 if request.form["loan_intent"] == "HOMEIMPROVEMENT" else 0,
                "loan_intent_MEDICAL": 1 if request.form["loan_intent"] == "MEDICAL" else 0,
                "loan_intent_PERSONAL": 1 if request.form["loan_intent"] == "PERSONAL" else 0,
                "loan_intent_VENTURE": 1 if request.form["loan_intent"] == "VENTURE" else 0,
                "loan_grade_B": 1 if request.form["loan_grade"] == "B" else 0,
                "loan_grade_C": 1 if request.form["loan_grade"] == "C" else 0,
                "loan_grade_D": 1 if request.form["loan_grade"] == "D" else 0,
                "loan_grade_E": 1 if request.form["loan_grade"] == "E" else 0,
                "loan_grade_F": 1 if request.form["loan_grade"] == "F" else 0,
                "loan_grade_G": 1 if request.form["loan_grade"] == "G" else 0,
                "cb_person_default_on_file_Y": 1 if request.form["cb_person_default_on_file"] == "Y" else 0
            }

            df = pd.DataFrame([input_data])

            # Ensure all expected features are present and ordered
            expected_cols = model.get_booster().feature_names
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected_cols]

            pred = model.predict(df)[0]
            proba = model.predict_proba(df)[0][1]

            prediction = {
                "result": "Approved" if pred == 0 else "Rejected",
                "probability": f"{proba * 100:.2f}% chance of default"
            }

        except Exception as e:
            prediction = {"error": str(e)}

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
