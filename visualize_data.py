from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns
import pandas as pd


def plot_target_pie_chart(df):
    df['Target'].value_counts().plot(kind='pie', autopct='%1.3f%%')
    plt.title('Pie chart of Target')
    plt.show()


def plot_distance_curricular_units(data, title):
    feature = 'Curricular units 2nd sem (approved)'
    plt.figure(figsize=(10, 6))
    sns.histplot(data.X[feature], bins=20, kde=True, color='blue')
    plt.title(f'Distribution of {feature} - {title}')
    plt.ylabel('Frequency')
    plt.show()


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
