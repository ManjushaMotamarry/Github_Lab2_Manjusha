# from sklearn.datasets import fetch_rcv1
import mlflow, datetime, os, pickle, random
# import sklearn
from joblib import dump
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
import sys
import argparse
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    
    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    X, y = load_digits(return_X_y=True)

    if os.path.exists('data'): 
        with open('data/data.pickle', 'wb') as data:
            pickle.dump(X, data)
            
        with open('data/target.pickle', 'wb') as data:
            pickle.dump(y, data)  
    else:
        os.makedirs('data/')
        with open('data/data.pickle', 'wb') as data:
            pickle.dump(X, data)
            
        with open('data/target.pickle', 'wb') as data:
            pickle.dump(y, data)  
            
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Digits Dataset"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"    
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                        run_name= f"{dataset_name}"):
        
        params = {
                    "dataset_name": dataset_name,
                    "number of datapoint": X.shape[0],
                    "number of dimensions": X.shape[1]}
        
        mlflow.log_params(params)
            
        
        model = GradientBoostingClassifier(random_state=0)
        model.fit(X, y)
        
        y_predict = model.predict(X)
        mlflow.log_metrics({'Accuracy': accuracy_score(y, y_predict),
                            'F1 Score': f1_score(y, y_predict)})
        
        if not os.path.exists('models/'): 
            # then create it.
            os.makedirs("models/")
            
        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = f'{model_version}_dt_model.joblib'
        print("Running GitHub Actions model training workflow...")
        dump(model, model_filename)
                    

