from dataclasses import dataclass
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score, \
    mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from visualize_data import plot_confusion_matrix, plot_feature_importance, plot_classification_report, plot_distance_heatmap, \
    plot_distance_heatmap_curricular_units, plot_residuals


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


