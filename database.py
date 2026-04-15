import pandas as pd
from sqlalchemy import create_engine
import os

class DatabaseManager:
    """Manages SQLite database connections and operations for financial data."""
    
    def __init__(self, db_path='sqlite:///finance_data.db'):
        """Initializes the database engine."""
        # Ensure directory exists if it's a file path
        if 'sqlite:///' in db_path:
            file_path = db_path.replace('sqlite:///', '')
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        self.engine = create_engine(db_path)

    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """Saves a pandas DataFrame to a database table.
        
        Args:
            df (pd.DataFrame): The DataFrame to save.
            table_name (str): The name of the table.
            if_exists (str): 'fail', 'replace', or 'append'.
        """
        try:
            df.to_sql(table_name, con=self.engine, if_exists=if_exists, index=False)
            print(f"Successfully saved {len(df)} rows to table '{table_name}'.")
        except Exception as e:
            print(f"Error saving DataFrame to table '{table_name}': {e}")

    def load_dataframe(self, table_name: str, query: str = None) -> pd.DataFrame:
        """Loads data from a database table into a pandas DataFrame.
        
        Args:
            table_name (str): The name of the table.
            query (str, optional): A custom SQL query. Defaults to None (loads entire table).
            
        Returns:
            pd.DataFrame: The loaded data.
        """
        try:
            if query:
                df = pd.read_sql(query, con=self.engine)
            else:
                df = pd.read_sql_table(table_name, con=self.engine)
            return df
        except Exception as e:
            print(f"Error loading DataFrame from table '{table_name}': {e}")
            return pd.DataFrame()

# Example usage (can be removed later)
if __name__ == "__main__":
    db = DatabaseManager()
    print("Database manager initialized.")