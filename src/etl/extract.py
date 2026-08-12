"""
Extract the Data from the data source that is csv file
"""

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import pandas as pd
from utils.logger import logging
from utils.exception import MyException
from utils.config_loader import settings


try:
    file_path = settings['data']['path']
except KeyError as e:
    logging.error(f"Missing 'data' or 'path' key in settings: {str(e)}")
    raise MyException(f"Configuration key missing: {str(e)}", sys)

if not file_path:
    raise MyException("File path is not defined in the configuration settings.", sys)


class ExtractData:
    
    def data_extraction(self) -> pd.DataFrame:
        """
        Loads the CSV data source into a pandas DataFrame.
        Returns:
            pd.DataFrame: The extracted dataset.
        """
        try:
            logging.info(f"Loading the data from the: {file_path}")
            
            # Extract data
            df = pd.read_csv(file_path)
            
            logging.info("Data loaded successfully.")
            logging.info(f"Shape of the data: {df.shape}")
            
            return df
            
        except FileNotFoundError as e:
            logging.error(f"The file was not found at the specified path: {file_path}")
            raise MyException(f"CSV File missing: {str(e)}", sys)
            
        except Exception as e:
            logging.error("An error occurred during data extraction.")
            raise MyException(e, sys)



# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    try:
        extractor = ExtractData()
        raw_data = extractor.data_extraction()
        print("First few rows:\n", raw_data.head())
    except MyException as e:
        print(f"Extraction failed: {e}")
