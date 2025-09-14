import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class Visualizer:
    def __init__(self, df=None):
        """
        Initialize the Visualizer class.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            DataFrame containing the data.
        """
        self.df = df
    
    def set_data(self, df):
        """
        Set the data for visualization.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the data.
        """
        self.df = df
    
    def plot_target_distribution(self):
        """
        Plot the distribution of the target variable.
        """
        if self.df is None:
            st.error("No data available for visualization.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='y', data=self.df, ax=ax)
        ax.set_title('Distribution of Target Variable (Term Deposit Subscription)')
        ax.set_xlabel('Subscribed to Term Deposit')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['No', 'Yes'])
        
        # Add percentage labels
        total = len(self.df)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 5,
                    '{:.1f}%'.format(100 * height/total),
                    ha="center") 
        
        st.pyplot(fig)
        
        # Display counts and percentages as a table
        target_counts = self.df['y'].value_counts()
        target_percent = self.df['y'].value_counts(normalize=True) * 100
        target_stats = pd.DataFrame({
            'Count': target_counts,
            'Percentage (%)': target_percent.round(2)
        })
        target_stats.index = ['No', 'Yes'] if 'no' in target_counts.index else target_counts.index
        
        st.write("### Target Distribution")
        st.dataframe(target_stats)
    
    def plot_categorical_features(self, columns=None):
        """
        Plot the distribution of categorical features.
        
        Parameters:
        -----------
        columns : list, optional
            List of categorical columns to plot. If None, all object columns are used.
        """
        if self.df is None:
            st.error("No data available for visualization.")
            return
        
        # If columns not specified, use all object columns
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
            # Exclude target variable if it's in the list
            if 'y' in columns:
                columns.remove('y')
        
        # Create subplots for each categorical feature
        for col in columns:
            st.write(f"### Distribution of {col}")
            
            # Get value counts and percentages
            val_counts = self.df[col].value_counts().sort_values(ascending=False)
            val_percent = self.df[col].value_counts(normalize=True).sort_values(ascending=False) * 100
            
            # Create DataFrame for display
            stats_df = pd.DataFrame({
                'Count': val_counts,
                'Percentage (%)': val_percent.round(2)
            })
            
            # Plot the distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(x=col, data=self.df, order=val_counts.index, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            
            # Rotate x-axis labels if there are many categories
            if len(val_counts) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display the statistics
            st.dataframe(stats_df)
            
            # Plot the relationship with target variable
            st.write(f"### {col} vs Target Variable")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate the percentage of positive class for each category
            # Create a temporary dataframe with numeric target
            temp_df = self.df.copy()
            temp_df['y_numeric'] = temp_df['y'].map({'yes': 1, 'no': 0})
            target_pct = temp_df.groupby(col)['y_numeric'].mean() * 100
            target_pct = target_pct.sort_values(ascending=False)
            
            # Create a new DataFrame for the plot
            plot_df = pd.DataFrame({
                'Category': target_pct.index,
                'Subscription Rate (%)': target_pct.values
            })
            
            # Plot the subscription rate for each category
            sns.barplot(x='Category', y='Subscription Rate (%)', data=plot_df, ax=ax)
            ax.set_title(f'Subscription Rate by {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Subscription Rate (%)')
            
            # Rotate x-axis labels if there are many categories
            if len(target_pct) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add a separator
            st.markdown("---")
    
    def plot_numerical_features(self, columns=None):
        """
        Plot the distribution of numerical features.
        
        Parameters:
        -----------
        columns : list, optional
            List of numerical columns to plot. If None, all numerical columns are used.
        """
        if self.df is None:
            st.error("No data available for visualization.")
            return
        
        # If columns not specified, use all numerical columns
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Exclude target variable if it's in the list
            if 'y' in columns:
                columns.remove('y')
        
        # Create subplots for each numerical feature
        for col in columns:
            st.write(f"### Distribution of {col}")
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            
            # Plot histogram
            sns.histplot(self.df[col], kde=True, ax=ax1)
            ax1.set_title(f'Histogram of {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Frequency')
            
            # Plot boxplot
            sns.boxplot(x=self.df[col], ax=ax2)
            ax2.set_title(f'Boxplot of {col}')
            ax2.set_xlabel(col)
            
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display basic statistics
            stats = self.df[col].describe()
            st.dataframe(stats)
            
            # Plot the relationship with target variable
            st.write(f"### {col} vs Target Variable")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create boxplot of the feature grouped by target
            sns.boxplot(x='y', y=col, data=self.df, ax=ax)
            ax.set_title(f'{col} by Target Variable')
            ax.set_xlabel('Subscribed to Term Deposit')
            ax.set_ylabel(col)
            ax.set_xticklabels(['No', 'Yes'])
            
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add a separator
            st.markdown("---")
    
    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix of numerical features.
        """
        if self.df is None:
            st.error("No data available for visualization.")
            return
        
        # Select numerical columns
        numerical_df = self.df.select_dtypes(include=['int64', 'float64'])
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Plot the correlation matrix
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax)
        ax.set_title('Correlation Matrix of Numerical Features')
        
        # Adjust layout
        plt.tight_layout()
        st.pyplot(fig)