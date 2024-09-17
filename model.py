import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import shap
from lime import lime_tabular
from pdpbox import pdp, info_plots

class CustomPipeline:
    def __init__(self):
        self.pipelines = {}
        self.models = {}
        self.results = {}
        self.feature_names = []

    def preprocess_data(self, data):
        X = data.drop(columns=['churn'])
        y = data['churn']
        self.feature_names = X.columns.tolist()
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def create_pipeline(self, X_train, model):
        numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = X_train.select_dtypes(exclude=['float64', 'int64']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
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
            pipeline = self.create_pipeline(X_train, model)
            param_grid = {
                'feature_selection__k': [5, 10, 15, 20, 'all']
            }
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            self.pipelines[model_name] = grid_search.best_estimator_
            self.results[model_name] = grid_search.best_score_

        best_model_name = max(self.results, key=self.results.get)
        return best_model_name, self.pipelines[best_model_name]

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def get_feature_importances(self, model):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = abs(model.coef_[0])
        return dict(zip(self.feature_names, importances))

    def generate_pdp_plots(self, model, X_test):
        pdp_plots = []
        for feature in self.feature_names[:5]:
            pdp_iso = pdp.pdp_isolate(model=model, dataset=X_test, model_features=self.feature_names, feature=feature)
            fig, ax = plt.subplots()
            pdp.pdp_plot(pdp_iso, feature, ax=ax)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            pdp_plots.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        return pdp_plots

    def generate_shap_summary_plot(self, model, X_train, X_test):
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_lime_explanation(self, model, X_train, X_test):
        explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=self.feature_names, class_names=['Not Churned', 'Churned'], mode='classification')
        exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba, num_features=10)
        return exp.as_html()

    def generate_surrogate_tree(self, model, X_train, y_train):
        surrogate_model = DecisionTreeClassifier(max_depth=3)
        surrogate_model.fit(X_train, model.predict(X_train))
        fig, ax = plt.subplots(figsize=(20,10))
        plot_tree(surrogate_model, feature_names=self.feature_names, filled=True, rounded=True, ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')