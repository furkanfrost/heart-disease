# data_loader.py
import pandas as pd
import os

def load_data(filename='heart.csv'):
    # Bu script hangi klasördeyse orayı temel al
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(df.head())
    print(df.info())
    return df

if __name__ == "__main__":
    df = load_data()
