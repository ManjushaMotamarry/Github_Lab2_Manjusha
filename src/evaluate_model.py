import pickle, os, json, random
from sklearn.metrics import f1_score
import joblib, glob, sys
import argparse
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}_dt_model'  # Use a timestamp as the version
        model = joblib.load(f'{model_version}.joblib')
    except:
        raise ValueError('Failed to catching the latest model')
        
    try:
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
    except Exception as e:
        raise ValueError(f'Failed to load digits dataset: {e}')
    
    y_predict = model.predict(X)
    metrics = {"F1_Score": f1_score(y, y_predict, average='weighted')}
    
    # Save metrics to a JSON file

    if not os.path.exists('metrics/'): 
        # then create it.
        os.makedirs("metrics/")
        
    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
               
    
