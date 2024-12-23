import os
from zipfile import BadZipFile
from flask import Flask, request, render_template
import pandas as pd
import openpyxl  # engine used in 'pd.read_excel'
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.utils import secure_filename

from ai_utils import iqr_processing, prepare_data, eval_classification
from predict_student_success import clean_data
from visualize_data import plot_target_pie_chart

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')

    if 'file' not in request.files:
        return 'No file part in the request', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    if file:
        try:
            ext = file.filename.split('.')[1]  # get file extension
            file_path = f'temp.{ext}'
            file.save(secure_filename(file_path))

            df = pd.read_excel(file_path, engine='openpyxl')
        except BadZipFile:
            filename = 'Predict Student Dropout and Academic Success.csv'
            if file.filename == filename:
                df = pd.read_csv(file_path, delimiter=';')
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            return render_template('index.html', error_message=e)

        try:
            results = process_and_get_visualized_data(df)
        except ValueError as e:
            return render_template('index.html', error_message=e)

    return render_template(
        'index.html',
        data=results['df'].to_html(classes='table table-striped'),
        outputs=results,
        plots=results['plots']
    )


def process_and_get_visualized_data(df):
    df = clean_data(df)
    outliers, iqr_output = iqr_processing(df)
    plot_target_pie_chart(df)
    data_splits = prepare_data(df)
    return {
        'df': df,
        'outliers': outliers.fillna('').to_html(classes='table table-striped'),
        'iqr_output': iqr_output.replace('\n', '<br>'),
        'lr_output': eval_classification(LogisticRegression(), data_splits, 'Logistic Regression'),
        'rf_output': eval_classification(RandomForestClassifier(), data_splits, 'Random Forest'),
        'knn_output': eval_classification(KNeighborsClassifier(), data_splits, 'KNN'),
        'describe': df.describe().to_html(classes='table table-striped'),
        'plots': [filename for filename in os.listdir(os.path.join('static', 'plots'))]
    }


if __name__ == '__main__':
    app.run(debug=True)
