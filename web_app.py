from zipfile import BadZipFile
from flask import Flask, request, render_template
import pandas as pd
import openpyxl  # engine used in 'pd.read_excel'

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

    return render_template('index.html', prediction=table_html)


if __name__ == '__main__':
    app.run(debug=True)
