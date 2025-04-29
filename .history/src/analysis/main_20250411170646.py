import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthcareAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the HealthcareAnalyzer with data path.
        
        Args:
            data_path (str): Path to the healthcare dataset
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the healthcare dataset."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, target_column):
        """
        Preprocess the data for analysis.
        
        Args:
            target_column (str): Name of the target variable column
        """
        try:
            logger.info("Starting data preprocessing")
            
            # Handle missing values
            self.data = self.data.fillna(self.data.mean())
            
            # Separate features and target
            self.X = self.data.drop(target_column, axis=1)
            self.y = self.data[target_column]
            
            # Scale features
            self.X = self.scaler.fit_transform(self.X)
            
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
            
    def train_model(self):
        """Train a Random Forest classifier on the preprocessed data."""
        try:
            logger.info("Training Random Forest model")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X, self.y)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target values
        """
        try:
            logger.info("Evaluating model performance")
            y_pred = self.model.predict(X_test)
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), 
                       annot=True, 
                       fmt='d',
                       cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
            
    def analyze_healthcare_trends(self):
        """Analyze and visualize healthcare trends in the data."""
        try:
            logger.info("Analyzing healthcare trends")
            
            # Example: Plot age distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x='age', bins=30)
            plt.title('Age Distribution of Patients')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.show()
            
            # Example: Plot correlation matrix
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
            plt.title('Feature Correlation Matrix')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the healthcare analysis workflow.
    """
    try:
        # Initialize analyzer
        analyzer = HealthcareAnalyzer('data/healthcare_dataset.csv')
        
        # Load and preprocess data
        analyzer.load_data()
        analyzer.preprocess_data(target_column='outcome')
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            analyzer.X, analyzer.y, test_size=0.2, random_state=42
        )
        
        # Train model
        analyzer.train_model()
        
        # Evaluate model
        analyzer.evaluate_model(X_test, y_test)
        
        # Analyze trends
        analyzer.analyze_healthcare_trends()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 