import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load data
data = pd.read_csv("student_data.csv")
X = data[['study_hours', 'attendance']]
y = data['pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])

        result = model.predict([[study_hours, attendance]])
        prediction = "PASS 🎉" if result[0] == 1 else "FAIL 📉"

    return render_template(
        "index.html",
        prediction=prediction,
        accuracy=round(accuracy * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
