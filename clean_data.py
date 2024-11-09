import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score, \
    mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_data(df):
    # Clean the column names (specifically 'Daytime/evening attendance')
    df.columns = df.columns.str.strip().str.replace('\t', '')
    df.columns = df.columns.str.replace('"', '')

    # Remove duplicates.
    df.drop_duplicates(inplace=True)

    # Change classified text under ‘Target’ column to be integers.
    df['Target'] = LabelEncoder().fit_transform(df['Target'])

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


def perform_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}\n')
    print(f'Confusion Matrix:\n{conf_matrix}\n')
    print(f'Classification Report:\n{class_report}\n')


def perform_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R²): {r2}')


def main():
    filename = "Predict Student Dropout and Academic Success.csv"
    df = pd.read_csv(filename, delimiter=';')
    clean_data(df)
    df.to_csv("cleaned_data.xlsx", index=False)

    iqr_processing(df)
    visualize_data(df)
    print(df.describe())

    X = df.drop(columns=['Target'])  # features
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    perform_logistic_regression(X_train, X_test, y_train, y_test)
    perform_linear_regression(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
