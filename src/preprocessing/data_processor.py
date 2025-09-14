import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class DataProcessor:
    def __init__(self, data_path=None, df=None):
        """
        Initialize the DataProcessor class.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing the data.
        df : pandas.DataFrame, optional
            DataFrame containing the data.
        """
        if data_path is not None:
            self.df = pd.read_csv(data_path, sep=';')
        elif df is not None:
            self.df = df.copy()
        else:
            self.df = None
            
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.label_encoders = {}
        
    def load_data(self, data_path):
        """
        Load data from a CSV file.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing the data.
        """
        self.df = pd.read_csv(data_path, sep=';')
        return self.df
    
    def get_data_info(self):
        """
        Get basic information about the data.
        
        Returns:
        --------
        dict
            Dictionary containing basic information about the data.
        """
        if self.df is None:
            return None
        
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'target_distribution': self.df['y'].value_counts().to_dict()
        }
        
        return info
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data for model training.
        
        Parameters:
        -----------
        test_size : float, optional
            Proportion of the data to be used as test set.
        random_state : int, optional
            Random state for reproducibility.
            
        Returns:
        --------
        tuple
            Tuple containing the preprocessed training and test sets.
        """
        if self.df is None:
            return None
        
        # Make a copy of the dataframe
        df = self.df.copy()
        
        # Convert target variable to binary (0/1)
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        
        # Split features and target
        X = df.drop('y', axis=1)
        y = df['y']
        
        # Split data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipelines for both numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Fit and transform the training data
        X_train_processed = self.preprocessor.fit_transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        return X_train_processed, X_test_processed, self.y_train, self.y_test
    
    def save_preprocessor(self, file_path):
        """
        Save the preprocessor to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the preprocessor.
        """
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, file_path)
            print(f"Preprocessor saved to {file_path}")
        else:
            print("Preprocessor not available. Run preprocess_data() first.")
    
    def load_preprocessor(self, file_path):
        """
        Load the preprocessor from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved preprocessor.
            
        Returns:
        --------
        ColumnTransformer
            The loaded preprocessor.
        """
        self.preprocessor = joblib.load(file_path)
        return self.preprocessor
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        input_data : pandas.DataFrame
            Input data to be preprocessed.
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed input data.
        """
        if self.preprocessor is None:
            print("Preprocessor not available. Load or create a preprocessor first.")
            return None
        
        # Convert input data to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Preprocess the input data
        processed_data = self.preprocessor.transform(input_data)
        
        return processed_data