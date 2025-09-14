import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ModelTrainer:
    def __init__(self):
        """
        Initialize the ModelTrainer class.
        """
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = {}
    
    def train_logistic_regression(self, X_train, y_train, **kwargs):
        """
        Train a logistic regression model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray or scipy.sparse matrix
            Training data.
        y_train : numpy.ndarray
            Target values.
        **kwargs : dict
            Additional parameters to pass to the model.
            
        Returns:
        --------
        sklearn.linear_model.LogisticRegression
            Trained logistic regression model.
        """
        # Set default parameters
        params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
        
        # Update parameters with provided kwargs
        params.update(kwargs)
        
        # Create and train the model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Store the model
        self.models['Logistic Regression'] = model
        
        return model
    
    def train_decision_tree(self, X_train, y_train, **kwargs):
        """
        Train a decision tree model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray or scipy.sparse matrix
            Training data.
        y_train : numpy.ndarray
            Target values.
        **kwargs : dict
            Additional parameters to pass to the model.
            
        Returns:
        --------
        sklearn.tree.DecisionTreeClassifier
            Trained decision tree model.
        """
        # Set default parameters
        params = {
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # Update parameters with provided kwargs
        params.update(kwargs)
        
        # Create and train the model
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        # Store the model
        self.models['Decision Tree'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """
        Train a random forest model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray or scipy.sparse matrix
            Training data.
        y_train : numpy.ndarray
            Target values.
        **kwargs : dict
            Additional parameters to pass to the model.
            
        Returns:
        --------
        sklearn.ensemble.RandomForestClassifier
            Trained random forest model.
        """
        # Set default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # Update parameters with provided kwargs
        params.update(kwargs)
        
        # Create and train the model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Store the model
        self.models['Random Forest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """
        Train an XGBoost model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray or scipy.sparse matrix
            Training data.
        y_train : numpy.ndarray
            Target values.
        **kwargs : dict
            Additional parameters to pass to the model.
            
        Returns:
        --------
        xgboost.XGBClassifier
            Trained XGBoost model.
        """
        # Set default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Update parameters with provided kwargs
        params.update(kwargs)
        
        # Create and train the model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Store the model
        self.models['XGBoost'] = model
        
        return model
    
    def train_lightgbm(self, X_train, y_train, **kwargs):
        """
        Train a LightGBM model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray or scipy.sparse matrix
            Training data.
        y_train : numpy.ndarray
            Target values.
        **kwargs : dict
            Additional parameters to pass to the model.
            
        Returns:
        --------
        lightgbm.LGBMClassifier
            Trained LightGBM model.
        """
        # Set default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Update parameters with provided kwargs
        params.update(kwargs)
        
        # Create and train the model
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # Store the model
        self.models['LightGBM'] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a model on the test set.
        
        Parameters:
        -----------
        model : object
            Trained model.
        X_test : numpy.ndarray or scipy.sparse matrix
            Test data.
        y_test : numpy.ndarray
            True target values.
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics.
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store metrics in a dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test, metric='f1_score'):
        """
        Evaluate all trained models on the test set.
        
        Parameters:
        -----------
        X_test : numpy.ndarray or scipy.sparse matrix
            Test data.
        y_test : numpy.ndarray
            True target values.
        metric : str, optional
            Metric to use for selecting the best model.
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics for all models.
        """
        results = {}
        
        for name, model in self.models.items():
            # Evaluate the model
            metrics = self.evaluate_model(model, X_test, y_test)
            results[name] = metrics
            
            # Update best model if current model is better
            if metrics[metric] > self.best_score:
                self.best_score = metrics[metric]
                self.best_model = model
                self.best_model_name = name
        
        self.results = results
        return results
    
    def plot_roc_curves(self, X_test, y_test):
        """
        Plot ROC curves for all trained models.
        
        Parameters:
        -----------
        X_test : numpy.ndarray or scipy.sparse matrix
            Test data.
        y_test : numpy.ndarray
            True target values.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the ROC curves.
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            # Get predicted probabilities
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--')
        
        # Set plot properties
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        
        return plt.gcf()
    
    def plot_confusion_matrix(self, model_name=None):
        """
        Plot confusion matrix for a specific model or the best model.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the model to plot confusion matrix for.
            If None, the best model is used.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the confusion matrix.
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_all_models() first.")
            return None
        
        # If model_name is not provided, use the best model
        if model_name is None:
            model_name = self.best_model_name
        
        # Get confusion matrix
        conf_matrix = self.results[model_name]['confusion_matrix']
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        
        return plt.gcf()
    
    def save_model(self, model=None, model_name=None, file_path=None):
        """
        Save a model to a file.
        
        Parameters:
        -----------
        model : object, optional
            Model to save. If None, the best model is used.
        model_name : str, optional
            Name of the model to save. If None, the best model name is used.
        file_path : str, optional
            Path to save the model. If None, a default path is used.
            
        Returns:
        --------
        str
            Path where the model was saved.
        """
        # If model is not provided, use the best model
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        
        # If file_path is not provided, use a default path
        if file_path is None:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            file_path = f'models/{model_name.lower().replace(" ", "_")}.pkl'
        
        # Save the model
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
        
        return file_path
    
    def load_model(self, file_path):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved model.
            
        Returns:
        --------
        object
            The loaded model.
        """
        model = joblib.load(file_path)
        return model