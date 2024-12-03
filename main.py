# This project currently outputs to terminal within functions


from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score, \
    mean_squared_error, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
    # chose mean because it's sensitive to every value. preserves the overall distribution and variance of the data.

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
    # 1.5 is a standard threshold that balances sensitivity to potential outliers with the risk of overfitting
    return outliers


def eval_classification(model, data, title):
    model.fit(data.X_train, data.y_train)
    y_pred = model.predict(data.X_test)

    accuracy = accuracy_score(data.y_test, y_pred)
    conf_matrix = confusion_matrix(data.y_test, y_pred)
    class_report = classification_report(data.y_test,
                                         y_pred,
                                         target_names=['Graduate', 'Dropout'],
                                         output_dict=True)

    plot_confusion_matrix(conf_matrix, title)
    print(f'Accuracy: {accuracy}\n')
    plot_classification_report(class_report, title)

    if isinstance(model, RandomForestClassifier):
        plot_feature_importance(model, data.X.columns, title)

    if isinstance(model, KNeighborsClassifier):
        plot_distance_heatmap(data, title)
        plot_distance_heatmap_curricular_units(data, title)

    # Save the trained model
    model_filename = f"{title.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_filename)


def plot_distance_heatmap(data, title):
    sns.heatmap(pairwise_distances(data.X_train), cmap='YlGnBu', annot=False)
    plt.title(f'Distance Matrix for Training Data {title}')
    plt.show()


def plot_distance_heatmap_curricular_units(data, title):
    feature = 'Curricular units 2nd sem (approved)'
    plt.figure(figsize=(10, 6))
    sns.histplot(data.X[feature], bins=20, kde=True, color='blue')
    plt.title(f'Distribution of {feature} - {title}')
    plt.ylabel('Frequency')
    plt.show()


def eval_regression(model, data, title):
    model.fit(data.X_train, data.y_train)
    y_pred = model.predict(data.X_test)

    mae = mean_absolute_error(data.y_test, y_pred)
    mse = mean_squared_error(data.y_test, y_pred)
    r2 = r2_score(data.y_test, y_pred)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R²): {r2}')

    plot_residuals(data.y_test, y_pred, title)

    # Save the trained model
    model_filename = f"{title.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_filename)


def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()


def plot_classification_report(report, title):
    report = pd.DataFrame(report).transpose()
    report = report.drop(columns=['support'])  # 'support' overwhelms other factors
    report = report.iloc[:-1, :].T

    plt.figure(figsize=(8, 6))
    sns.heatmap(report, annot=True, cmap="YlGnBu", cbar=False, fmt=".2f")
    plt.title(f'Classification Report Heatmap - {title}')
    plt.show()


def plot_residuals(y_test, y_pred, title):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {title}')
    plt.show()


def plot_feature_importance(model, feature_names, title):
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    feature_importances = feature_importances.sort_values(by='Importance')

    plt.figure(figsize=(12, 10))
    plt.title(f'Feature Importances - {title}')
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='Blues_d', hue='Feature')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.subplots_adjust(left=0.3)
    plt.show()


def plot_first_tree(model, data, title):
    plt.figure(figsize=(20, 10))
    class_names = list(map(str, data.y.unique()))
    plot_tree(model.estimators_[0], filled=True, feature_names=data.X.columns, class_names=class_names)
    plt.title(f'First Decision Tree in Random Forest - {title}')
    plt.show()


def main():
    filename = "Predict Student Dropout and Academic Success.csv"
    df = pd.read_csv(filename, delimiter=';')
    df = clean_data(df)
    df.to_csv("cleaned_data.xlsx", index=False)

    outliers = iqr_processing(df)
    outliers.to_csv("outliers.xlsx", index=False)
    visualize_data(df)
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


# Decision trees are highly sensitive to training data, resulting in high variance.
#
# “forest” because it uses multiple trees
#
# randomly build multiple new datasets by randomly (“random”) selecting data
# called bootstrapping
# ensures we don’t use same data for every tree
#
# aggregation - combining predictions from multiple decision trees - majority voting
#
# last two together is called bagging