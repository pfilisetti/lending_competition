import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


def load_data(past_loans_path='data/PastLoans.csv', new_apps_path='data/NewApplications_Lender1_Round1.csv'):
    """Loads the past loans and new applications data from CSV files."""
    try:
        past_loans_df = pd.read_csv(past_loans_path)
        new_applications_df = pd.read_csv(new_apps_path)
        print("Data loaded successfully.")
        return past_loans_df, new_applications_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the CSV files are in the correct path.")
        return None, None

def split_data(df, ratio, target_col='default'):
    """
    Splits the dataframe into two parts based on the given ratio.
    """
    # Stratify by the target column to ensure the distribution of the target variable 
    # is the same in both resulting dataframes.
    df_1, df_2 = train_test_split(
        df, test_size=ratio, random_state=42, stratify=df[target_col]
    )
    
    print(f"Data split into df_1 ({df_1.shape[0]} rows) and df_2 ({df_2.shape[0]} rows).")
    return df_1, df_2

def create_preprocessor(categorical_features):
    """Creates a ColumnTransformer to preprocess categorical features."""
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

def grid_search(X, y, pipeline):
    """
    Performs a grid search to find the best hyperparameters for the XGBoost model.
    """
    print("Starting grid search for optimal hyperparameters...")
    
    # Define a smaller, efficient parameter grid for XGBoost
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__subsample': [0.7, 1.0],
        'classifier__colsample_bytree': [0.7, 1.0]
    }
    
    # Use Stratified K-Folds for cross-validation as the target is likely imbalanced
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Set up GridSearchCV
    search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    search.fit(X, y)
    
    print(f"\nBest hyperparameters found: {search.best_params_}")
    print(f"Best ROC AUC score from grid search: {search.best_score_:.4f}")
    
    # Return only the classifier's best parameters
    best_params = {key.replace('classifier__', ''): value for key, value in search.best_params_.items()}
    return best_params

def train_model(loans_df, features):
    """
    Trains an XGBoost model using cross-validation and hyperparameter tuning.
    """
    print(f"\n--- Training XGBoost model for features: {features} ---")
    
    # Automatically identify categorical features from the provided feature list
    categorical_features = loans_df[features].select_dtypes(include=['object']).columns.tolist()
    print(f"Identified categorical features: {categorical_features}")

    # Create a preprocessor specifically for this model instance
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X = loans_df[features]
    y = loans_df['default']
    
    # Create a base XGBoost classifier
    # 'eval_metric' is set to suppress a warning
    xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    
    # Create the full pipeline using the locally created preprocessor
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', xgb_classifier)])

    # Find the best hyperparameters
    best_hyperparams = grid_search(X, y, pipeline)
    
    # Create a new classifier with the best hyperparameters
    optimized_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42, **best_hyperparams)
    
    # Rebuild the pipeline with the optimized classifier and the same preprocessor instance
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', optimized_classifier)])

    # Evaluate the final model using cross-validation
    print("\nEvaluating final model with 5-fold cross-validation...")
    cv_scores = cross_val_score(final_pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC AUC scores: {np.round(cv_scores, 4)}")
    print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train the final model on the entire dataset
    print("\nTraining final model on all available data...")
    final_pipeline.fit(X, y)
    print("Model training complete.")
    
    return final_pipeline

def apply_prediction(df, model, features):
    """
    Applies a trained model to a dataframe and adds the prediction as a new column.
    """
    X = df[features]
    
    # Get the probability of the positive class (default=1)
    predictions = model.predict_proba(X)[:, 1]
    
    # Create a copy to avoid modifying the original dataframe
    df_predicted = df.copy()
    df_predicted['prediction'] = predictions
    
    return df_predicted