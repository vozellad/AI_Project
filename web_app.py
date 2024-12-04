from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('logistic_regression_model.pkl')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', prediction=None)
    # Get form data from user
    data = [float(request.form['feature1']), float(request.form['feature2'])]
    input_data = np.array(data).reshape(1, -1)
    prediction = model.predict(input_data)
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
