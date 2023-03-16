import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re

class CrossTermComputer:
    DIVIDER = ' x '

    def __init__(self, first_columns, second_columns, column_pairs):
        self.first_columns = first_columns
        self.second_columns = second_columns
        self.column_pairs = column_pairs
        self.column_names = self.compute_column_names()

    def compute_column_names(self):
        column_names_set = set()
        for col1 in self.first_columns:
            for col2 in self.second_columns:
                [alph1, alph2] = sorted([col1, col2])
                col_name = alph1 + ' x ' + alph2
                column_names_set.add(col_name)
        for (col1, col2) in self.column_pairs:
            [alph1, alph2] = sorted([col1, col2])
            col_name = alph1 + CrossTermComputer.DIVIDER + alph2
            column_names_set.add(col_name)
        column_names = sorted(list(column_names_set))
        return column_names

    def compute_cross_terms(self, df):
        ret_df = pd.DataFrame()
        for col_name in self.column_names:
            [col1, col2] = col_name.split(CrossTermComputer.DIVIDER)
            ret_df[col_name] = df[col1] * df[col2]
        return ret_df


class ZipcodeEncoder:
    def __init__(self):
        self.encoder = None
        self.dummy_zips_df = None
    
    # Actually this is unneeded since we already extracted zip codes :-/
    def extract_zip(self, address):
        zip_re = r'.*\, SC (2[0-9]{4})$'
        m = re.match(zip_re, address)
        return m.group(1)

        
    def generate_zipcode_dummy_vars(self, df, min_zipcode_count=10):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output = False, min_frequency=min_zipcode_count)
        nzips = df.shape[0]
        Dummy_Zips = self.encoder.fit_transform(df.zip.to_numpy().reshape(nzips, 1))
        col_names = self.encoder.get_feature_names_out()
        col_names = [name.replace('x0', 'zip') for name in col_names]
        infrequent_col_names = [name for name in col_names if name.find('infrequent') != -1]
        zip_df = pd.DataFrame(Dummy_Zips, columns=col_names)
        zip_df.drop(columns=infrequent_col_names, inplace=True)
        self.dummy_zips_df = zip_df
    
