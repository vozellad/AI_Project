# This project currently outputs to terminal within functions
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score, \
    mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree


@dataclass
class DataSplits:
    X: any
    y: any
    X_train: any
    X_test: any
    y_train: any
    y_test: any


def prepare_data(df):
    X = df.drop(columns=['Target'])  # features
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return DataSplits(X, y, X_train, X_test, y_train, y_test)


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

    # Set proper precision.
    df = df.round(3)

    return df


def visualize_data(df):
    df['Target'].value_counts().plot(kind='pie', autopct='%1.3f%%')
    plt.title('Pie chart of Target')
    plt.show()


def iqr_processing(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    print(f'Variance:\n{df.var()}\n')
    print(f'Standard Deviation:\n{df.std()}\n')
    print(f'Inter-quartile Range:\n{iqr}\n\n')

    outliers = df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))]
    print(f'Outliers:\n{outliers}\n\n')
    outliers.to_csv("outliers.xlsx", index=False)


def eval_classification(model, data):
    model.fit(data.X_train, data.y_train)
    y_pred = model.predict(data.X_test)

    accuracy = accuracy_score(data.y_test, y_pred)
    conf_matrix = confusion_matrix(data.y_test, y_pred)
    class_report = classification_report(data.y_test, y_pred)

    print(f'Accuracy: {accuracy}\n')
    print(f'Classification Report:\n{class_report}\n')

    if isinstance(model, RandomForestClassifier):
        plot_feature_importance(model, data.X.columns)
        plot_first_tree(model, data)


def eval_regression(model, data):
    model.fit(data.X_train, data.y_train)
    y_pred = model.predict(data.X_test)

    mae = mean_absolute_error(data.y_test, y_pred)
    mse = mean_squared_error(data.y_test, y_pred)
    r2 = r2_score(data.y_test, y_pred)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R²): {r2}')

    plot_residuals(data.y_test, y_pred)


def plot_confusion_matrix(matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot for Linear Regression')
    plt.show()


def plot_feature_importance(model, feature_names):
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    feature_importances = feature_importances.sort_values(by='Importance')

    plt.figure(figsize=(12, 10))
    plt.title('Feature Importances')
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='Blues_d', hue='Feature')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.subplots_adjust(left=0.3)
    plt.show()



def main():
    filename = "Predict Student Dropout and Academic Success.csv"
    df = pd.read_csv(filename, delimiter=';')
    df = clean_data(df)
    df.to_csv("cleaned_data.xlsx", index=False)

    iqr_processing(df)
    visualize_data(df)
    print(df.describe())

    data_splits = prepare_data(df)

    eval_classification(LogisticRegression(), data_splits)
    eval_regression(LinearRegression(), data_splits)
    eval_classification(RandomForestClassifier(), data_splits)


if __name__ == '__main__':
    main()
