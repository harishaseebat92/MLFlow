import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(df[["rm"]], df["medv"], test_size=0.33, random_state=42)

# Define models for regression
models = [
    (
        "Linear Regression", 
        LinearRegression(), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest", 
        RandomForestRegressor(n_estimators=100, random_state=42), 
        (X_train, y_train),
        (X_test, y_test)
    )
]

# Set up MLflow experiment
mlflow.set_experiment("LR_RF")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# List to store performance reports
reports = []

# Train and log each model
for model_name, model, train_set, test_set in models:
    X_train, y_train = train_set
    X_test, y_test = test_set
    
    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics for regression
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics, parameters, and models in MLflow
    with mlflow.start_run(run_name=model_name):        
        # Log model name as a parameter
        mlflow.log_param("model", model_name)
        
        # Log regression metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Infer the model signature
        signature = infer_signature(X_train, y_pred)
        
        # Log the model with its metadata
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_name}_model",
            signature=signature,
            input_example=X_train.head()
        )

        # Append report to list for later inspection if needed
        reports.append({"model_name": model_name, "mse": mse, "r2_score": r2})

# Print out the performance reports for each model
for report in reports:
    print(f"Model: {report['model_name']}, MSE: {report['mse']:.2f}, R²: {report['r2_score']:.2f}")
