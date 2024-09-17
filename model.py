import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class CustomPipeline:
    def __init__(self):
        self.pipelines = {}
        self.models = {}
        self.results = {}
        self.best_model = None

    def preprocess_data(self, data):
        X = data.drop(columns=['churn'])
        y = data['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def create_pipeline(self, X_train, model):
        numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = X_train.select_dtypes(exclude=['float64', 'int64']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
            ('scaler', StandardScaler())  # Normalize numeric data
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Convert categorical data to numeric using one-hot encoding
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        feature_selector = SelectKBest(f_classif)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', feature_selector),
            ('classifier', model)
        ])
        
        return pipeline

    def train_model(self, X_train, y_train):

        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            pipeline = self.create_pipeline(X_train, model)
           
            param_grid = {
                'feature_selection__k': ['all', 5, 10, 15]
            }
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_pipeline = grid_search.best_estimator_
            best_score = grid_search.best_score_
            
            self.pipelines[model_name] = best_pipeline
            self.results[model_name] = best_score

    def evaluate_model(self, X_test, y_test):

        best_model_name = max(self.results, key=self.results.get)
        best_model = self.pipelines[best_model_name]

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "best_model": best_model_name,
            "accuracy": accuracy,
            "classification_report": report
        }

    def save_best_model(self, file_path='best_model.pkl'):
        with open(file_path, 'wb') as file:
            pickle.dump(self.best_model, file)
        return file_path
