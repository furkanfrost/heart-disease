# data_loader.py
import pandas as pd
import os

def load_data(filename='heart.csv'):
    """
    Load data from the specified CSV file.
    
    Args:
        filename (str): Name of the CSV file to load
        
    Returns:
        pandas.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
        pd.errors.ParserError: If there's an error parsing the CSV file
    """
    # Use the directory where this script is located as the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', filename)

    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_csv(data_path)
        
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")
            
        print("Data loaded successfully:")
        print(df.head())
        print("\nDataFrame Info:")
        print(df.info())
        return df
        
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        raise

if __name__ == "__main__":
    try:
        df = load_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
