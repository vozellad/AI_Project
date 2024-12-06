import os
from zipfile import BadZipFile
from flask import Flask, request, render_template
import pandas as pd
import openpyxl  # engine used in 'pd.read_excel'
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from ai_utils import iqr_processing, prepare_data, eval_classification
from predict_student_success import clean_data
from visualize_data import plot_target_pie_chart

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    table_html = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in the request', 400
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        if file:
            try:
                df = pd.read_excel('temp.xlsx', engine='openpyxl')
            except BadZipFile:
                df = pd.read_csv('temp.xlsx')
            table_html = df.to_html(classes='table table-striped')
            process_and_get_visualized_data(df)

    return render_template('index.html', prediction=table_html)


def process_and_get_visualized_data(df):
    df = clean_data(df)
    outliers, output = iqr_processing(df)
    plot_target_pie_chart(df)
    describe = df.describe()
    data_splits = prepare_data(df)
    lr_output = eval_classification(LogisticRegression(), data_splits, 'Logistic Regression')
    rf_output = eval_classification(RandomForestClassifier(), data_splits, 'Random Forest')
    knn_output = eval_classification(KNeighborsClassifier(), data_splits, 'KNN')
    return {
        "output": output,
        "lr_output": lr_output,
        "rf_output": rf_output,
        "knn_output": knn_output,
        "describe": describe.to_html(classes='table table-striped'),
        "plots": [f"plots/{filename}" for filename in os.listdir("plots")]
    }


if __name__ == '__main__':
    app.run(debug=True)
