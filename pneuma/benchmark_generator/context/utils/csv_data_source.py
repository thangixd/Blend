import os
import pandas as pd


class CsvDataSource:
    def __init__(self, data_source: str):
        """
        Initialize the representation of a csv data source

        ### Parameters:
        - data_source (str): path to the data source with CSV files inside.
        """
        self._data_source = data_source
        self.csv_file_names = [
            f for f in sorted(os.listdir(self._data_source)) if f.endswith(".csv")
        ]

    def __iter__(self):
        self.pointer = 0
        return self

    def __next__(self):
        """
        Return the file name and the contents.
        """
        if self.pointer >= len(self.csv_file_names):
            raise StopIteration

        csv_file_name = f"{self._data_source}/{self.csv_file_names[self.pointer]}"
        df = pd.read_csv(csv_file_name, on_bad_lines="skip")

        content = "col: " + " | ".join(df.columns)
        self.pointer += 1
        return (self.csv_file_names[self.pointer-1][:-4], content, len(df))

    def set_data_source(self, data_source: str):
        """
        Re-assign the data source
        """
        self._data_source = data_source
        self.csv_file_names = [
            f for f in os.listdir(self._data_source) if f.endswith(".csv")
        ]


if __name__ == "__main__":
    csv_data_source = CsvDataSource("data_src/tables/pneuma_adventure_works")
    csv_iterator = iter(csv_data_source)
    information = next(csv_iterator)
    print(information)
