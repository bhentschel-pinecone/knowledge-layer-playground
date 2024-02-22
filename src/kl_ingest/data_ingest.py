import pandas
import os
from typing import List

class DataLoader:

    def parquet_folder_to_pandas(self, folder: str) -> List[pandas.DataFrame]:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        files = files[:1]
        return [self.parquet_file_to_pandas(os.path.join(folder, f)) for f in files]

    def parquet_file_to_pandas(self, file: str) -> pandas.DataFrame:
        """
        Load a parquet file into a pandas DataFrame

        Args:
            file: Path to parquet file

        Returns:
            pandas.DataFrame: DataFrame containing the parquet data
        """
        return pandas.read_parquet(file)

    def csv_file_to_pandas(self, file: str) -> pandas.DataFrame:
        """
        Load a csv file into a pandas DataFrame

        Args:
            file: Path to csv file

        Returns:
            pandas.DataFrame: DataFrame containing the csv data
        """
        return pandas.read_csv(file)