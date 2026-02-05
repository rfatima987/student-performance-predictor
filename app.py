import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and train model
data = pd.read_csv("student_data.csv")
X = data[['study_hours', 'attendance']]
y = data['pass']

model = LogisticRegression()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])

        result = model.predict([[study_hours, attendance]])
        prediction = "PASS 🎉" if result[0] == 1 else "FAIL 📉"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
