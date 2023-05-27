import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    ############################### manual
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # mlflow.set_experiment("nyc-taxi-experiment")

    # X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    # X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    # with mlflow.start_run():
    #     mlflow.set_tag("developer", "UD")
        
    #     mlflow.log_param("train-data-pkl_path", os.path.join(data_path, "train.pkl"))
    #     mlflow.log_param("valid-data-pkl_path", os.path.join(data_path, "val.pkl"))

    #     depth = 10
    #     mlflow.log_param("depth", depth)
        
    #     rf = RandomForestRegressor(max_depth=depth, random_state=0)
    #     rf.fit(X_train, y_train)
        
    #     y_pred_train = rf.predict(X_train)
    #     y_pred = rf.predict(X_val)

    #     rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    #     rmse = mean_squared_error(y_val, y_pred, squared=False)
        
    #     mlflow.log_metric("train_rmse", rmse_train)
    #     mlflow.log_metric("val_rmse", rmse)
        
    # print(f"Validation RMSE: {rmse}")
    
    ############################### autolog - recommended
    mlflow.sklearn.autolog() # RECOMMENDED
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)


if __name__ == '__main__':
    run_train()
