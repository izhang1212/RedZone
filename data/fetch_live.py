import sqlite3
import pandas as pd
from config import DB_PATH

# Get connection to the SQLite db
def get_connection():
    try:
        connect = sqlite3.connect(DB_PATH)
        return connect
    
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

# Saves pandas dataframe to the sqlite db
def save_dataframe(df, table_name, if_exists='replace'):
    connect = get_connection()

    if connect:
        try:
            df.to_sql(table_name, connect, if_exists=if_exists, index=False)
            print(f"Successfully saved {len(df)} rows to table '{table_name}'.")
        
        except Exception as e:
            print(f"Error saving to table {table_name}: {e}")
        
        finally:
            connect.close()

# Executes a SQL query and returns the result as a pandas dataframe
def run_query(query):
    connect = get_connection()
    if connect:
        try:
            df = pd.read_sql_query(query, connect)
            return df
        
        except Exception as e:
            print(f"Error running query: {e}")
            return pd.DataFrame()
        
        finally:
            connect.close()
    
    return pd.DataFrame()