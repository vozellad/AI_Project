# This project currently outputs to terminal within functions

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from ai_utils import iqr_processing, prepare_data, eval_classification
from visualize_data import plot_target_pie_chart


def clean_data(df):
    # Clean the column names (specifically 'Daytime/evening attendance')
    df.columns = df.columns.str.strip().str.replace('\t', '')
    df.columns = df.columns.str.replace('"', '')

    # Remove duplicates.
    df.drop_duplicates(inplace=True)

    # This project is analyzing whether they dropped out or graduated.
    # If they’re still taking classes, their data is incomplete.
    df = df[df['Target'] != 'Enrolled']

    # Change classified text under ‘Target’ column to be integers.
    pd.set_option('future.no_silent_downcasting', True)
    df.loc[:, 'Target'] = df['Target'].replace({'Dropout': 0, 'Graduate': 1})

    # Strip leading and trailing white-spaces.
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop empty rows.
    df.dropna(how='all', inplace=True)

    # Replace missing values with the mean.
    df.fillna(df.mean(), inplace=True)
    # chose mean because it's sensitive to every value. preserves the overall distribution and variance of the data.

    # Set proper precision.
    df = df.round(3)

    return df


def main():
    filename = "Predict Student Dropout and Academic Success.csv"
    df = pd.read_csv(filename, delimiter=';')
    df = clean_data(df)
    df.to_csv("cleaned_data.xlsx", index=False)

    outliers = iqr_processing(df)
    outliers.to_csv("outliers.xlsx", index=False)
    plot_target_pie_chart(df)
    describe = df.describe()
    describe.to_csv('describe_data.xlsx')

    data_splits = prepare_data(df)

    print('LOGISTIC REGRESSION')
    eval_classification(LogisticRegression(), data_splits, 'Logistic Regression')
    print('RANDOM FOREST')
    eval_classification(RandomForestClassifier(), data_splits, 'Random Forest')
    print('KNN')
    eval_classification(KNeighborsClassifier(), data_splits, 'KNN')


if __name__ == '__main__':
    main()
