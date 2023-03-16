import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

import re


import imp
import bullets
imp.reload(bullets)
from blurbs import BlurbFeatures
from bullets import BulletFeatures



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
        print("columns in dataframe: ", df.columns)
        print(df['units_in_building'])
        print("In compute cross terms")
        ret_df = pd.DataFrame()
        for col_name in self.column_names:
            [col1, col2] = col_name.split(CrossTermComputer.DIVIDER)
            print("col1: ", col1, " col2: ", col2)
            print(df[col1].dtype)
            print(df[col2].dtype)
            ret_df[col_name] = df[col1] * df[col2]
        return ret_df


class ZipcodeEncoder:
    def __init__(self):
        self.encoder = None
        self.training_dummy_zips_df = None
        self.col_names = None
    
    # Actually this is unneeded since we already extracted zip codes :-/
    def extract_zip(self, address):
        zip_re = r'.*\, SC (2[0-9]{4})$'
        m = re.match(zip_re, address)
        return m.group(1)

    def _convert_results_to_df(self, dummy_zips_np):
        col_names = self.encoder.get_feature_names_out()
        col_names = [name.replace('x0', 'zip') for name in col_names]
        infrequent_col_names = [name for name in col_names if name.find('infrequent') != -1]
        zip_df = pd.DataFrame(dummy_zips_np, columns=col_names)
        print("zip df shape before dropping infrequent cols ", zip_df.shape)
        zip_df.drop(columns=infrequent_col_names, inplace=True)
        print("zip df shape after dropping infrequent cols ", zip_df.shape)
        return zip_df
        
    def generate_training_zipcode_dummy_vars(self, df, min_zipcode_count=10):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output = False, min_frequency=min_zipcode_count)
        Dummy_Zips = self.encoder.fit_transform(df[['zip']])  # note that here we do fit_transform


        #Dummy_Zips = self.encoder.fit_transform(df.zip.to_numpy().reshape(n, 1))  # note that here we do fit_transform
        zips_df = self._convert_results_to_df(Dummy_Zips)
        self.training_dummy_zips_df = zips_df
        return zips_df

    
    def generate_testing_zipcode_dummy_vars(self, df):
        n = df.shape[0]
        Dummy_Zips = self.encoder.transform(df.zip.to_numpy().reshape(n, 1))  # note that here we just transform - encoder was fit on training data
        zips_df = self._convert_results_to_df(Dummy_Zips)
        print("zips_df.shape ", zips_df.shape)
        return zips_df


# splits apt_df and addr_df into training, cross-validation, and testing subsets.
# split is done by the building - a building is only in one of the subsets, so all of its units will be together.
def split_data(apt_df, addr_df):
    apt_df_shape = apt_df.shape
    addr_df_shape = addr_df.shape

    # Drop rows with missing data in apt_df
    drop_apt_row = apt_df.isna().any(axis=1)
    apt_df = apt_df.loc[~drop_apt_row]
    apt_df.reset_index(inplace=True, drop=True)

    # addr_df shouldn't have missing data
    addr_rows_missing_data = addr_df.isna().any(axis=1)
    # Once we've gotten all the latlngs and cleaned the address data, add this line back:
    # assert(addr_rows_missing_data.sum() == 0)
    addr_df = addr_df.loc[~addr_rows_missing_data]

    # But it might have addresses listed twice
    addr_df.drop_duplicates(subset=['address'], inplace=True)
    addr_df.reset_index(inplace=True, drop=True)

    print("apt_df shape before: ", apt_df_shape)
    print('addr_df shape before: ', addr_df_shape)
    print("apt_df shape after: ", apt_df.shape)
    print('addr_df shape after: ', addr_df.shape)


    num_addrs = addr_df.shape[0]
    addr_df = shuffle(addr_df).reset_index(drop=True)
    num_train = int(0.65 * num_addrs)
    num_val = int(0.2 * num_addrs)
    num_test = num_addrs - (num_train + num_val)
    train_indices = pd.Series([True for i in range(num_train)] + [False for i in range(num_addrs - num_train)])
    val_indices = pd.Series([False for i in range(num_train)] + [True for i in range(num_val)] + [False for i in range(num_test)])
    test_indices = ~(train_indices | val_indices)
    addr_df_train = addr_df.loc[train_indices].reset_index(drop=True)
    addr_df_validation = addr_df.loc[val_indices].reset_index(drop=True)
    addr_df_test = addr_df.loc[test_indices].reset_index(drop=True)

    # Fuck all this for now. Let's just split the addresses 65:20:15 and go for it.
    # TODO: fix this code to split on number of units, to ensure balanced split.
    # 
    # redefine this to account for missing data
    # apt_df['units_in_building'] = 
    # # add the unit count to the addr_df, so that we can split the buildings up but count how many units are put into each dataset.
    # addrs_and_unit = apt_df[['address', 'units_in_building']].drop_duplicates()
    # addr_with_unit_count = addr_df.merge(addrs_and_unit, on='address').sort_values(by=['units_in_building'])
    # print('address_with_unit_count.shape ', addr_with_unit_count.shape)
    # num_units = apt_df.shape[0]
    # min_training_units = int(0.7 * num_units)
    # min_validation_units = int(0.2 * num_units)
    
    # print('min training units, min val units, sum', min_training_units, min_validation_units, min_training_units + min_validation_units)
    # print('num total units: ', num_units)

    # addr_with_unit_count = shuffle(addr_with_unit_count)
    # addr_with_unit_count.reset_index(inplace=True, drop = True)
    # addr_with_unit_count['unit_sum'] = addr_with_unit_count.units_in_building.cumsum()

    # training_indices = addr_with_unit_count.unit_sum <= min_training_units
    # val_indices = (addr_with_unit_count.unit_sum > min_training_units) & (addr_with_unit_count.unit_sum <= min_training_units + min_validation_units)
    # test_indices = ~(training_indices | val_indices)
    # print("indices shape: ", val_indices.shape)

    # print("num training indices:", training_indices.sum())
    # print("num val indices", val_indices.sum())
    # print("num test indices", test_indices.sum())

    # training_indices.to_csv('/tmp/train_indices.csv')
    # val_indices.to_csv('/tmp/val_indices.csv')

    # addr_df_train = addr_with_unit_count.loc[addr_with_unit_count.unit_sum <= min_training_units]
    # addr_df_validation = addr_with_unit_count.loc[(addr_with_unit_count.unit_sum > min_training_units) & (addr_with_unit_count.unit_sum <= min_training_units + min_validation_units)].reset_index(drop=True)
    # # addr_df_test = addr_with_unit_count.loc[addr_with_unit_count.unit_sum > (min_training_units + min_validation_units)].reset_index(drop=True)
    # addr_df_test = addr_with_unit_count.loc[test_indices].reset_index(drop=True)
    # # addr_df_train.drop(columns=['units_in_building', 'unit_sum'], inplace = True)
    # # addr_df_validation.drop(columns=['units_in_building', 'unit_sum'], inplace = True)
    # # addr_df_test.drop(columns=['units_in_building', 'unit_sum'], inplace = True)


    apt_df_train = pd.merge(apt_df, addr_df_train[['address']], on='address')
    apt_df_validation = pd.merge(apt_df, addr_df_validation[['address']], on='address')
    apt_df_test = pd.merge(apt_df, addr_df_test[['address']], on='address')

    return ((apt_df_train, addr_df_train), (apt_df_validation, addr_df_validation), (apt_df_test, addr_df_test))





class FeatureGenerator:
    def __init__(self):
        self.cross_term_computer = None
        self.blurb_features = BlurbFeatures()
        self.zip_encoder = ZipcodeEncoder()
        self.bullet_features = BulletFeatures(min_bullet_count = 20, num_svd_components = 20)
    
    def _merge_svd_feats_and_apt_df(self, svd_df, X, addr_df):
        # we have to concat these, instead of merge them, because scikitlearn just returns a numpy array.
        addrs_and_blurb_svd_feats = pd.concat((addr_df, svd_df), axis = 1)
        addrs_and_blurb_svd_feats.drop(columns=['blurb', 'url', 'title', 'bullets', 'latlng'], inplace=True)
        addrs_and_blurb_svd_feats.drop_duplicates(subset=['address'], inplace=True)
        X_with_blurbs = X.merge(addrs_and_blurb_svd_feats, on='address')
        return X_with_blurbs
    
    def _extract_lat_and_lng(self, latlng):
        latlng_list = re.sub(r'[\(\)]', '', latlng).split(',')
        [lat, lng] = map(lambda s : float(s.strip()), latlng_list)
        return (lat, lng)

    def _add_lat_long_features(self, apt_df, addr_df):
        addrs_and_latlng = addr_df[['address', 'latlng']]
        addrs_and_latlng.drop_duplicates(inplace=True)

        addrs_and_latlng['lat'] = addrs_and_latlng.latlng.map(lambda s : self._extract_lat_and_lng(s)[0])
        addrs_and_latlng['lng'] = addrs_and_latlng.latlng.map(lambda s : self._extract_lat_and_lng(s)[1])
        addrs_and_latlng['lat^2'] = addrs_and_latlng.lat ** 2
        addrs_and_latlng['lng^2'] = addrs_and_latlng.lng ** 2
        addrs_and_latlng['lat * lng'] = addrs_and_latlng.lat * addrs_and_latlng.lng
        addrs_and_latlng.drop(columns=['latlng'], inplace=True)
        X = apt_df.merge(addrs_and_latlng, on='address')
        return X
    
    def _add_property_type(self, X, addr_df):
        def type_from_title(title):
            m = re.match(r'.* ([A-Za-z]+) for Rent.*', title)
            if m is None:
                return 'undefined'
            type = m.group(1).lower()
            if type == 'houses':
                type = 'house'
            return type
        addr_df['property_type'] = addr_df.title.map(type_from_title)
        addr_and_type = addr_df[['property_type', 'address']]
        addr_and_type = pd.concat((addr_and_type, pd.get_dummies(addr_and_type.property_type)), axis=1)
        X_with_type = X.merge(addr_and_type, on='address')
        return X_with_type


    
    def _merge_bullet_svd_and_apt_df(self, bullet_svd_df, apt_df, addr_df):
        addrs = addr_df[['address']]
        addrs_and_bullet_svd_feats = addrs.join(bullet_svd_df)
        assert(addrs_and_bullet_svd_feats.shape[0] == bullet_svd_df.shape[0])
        X_with_bullets = apt_df.merge(addrs_and_bullet_svd_feats, on='address')
        return X_with_bullets

    def get_training_features(self, apt_df_train, addr_df_train):
        X0 = apt_df_train.sort_values(by=['address']).reset_index(drop=True)
        print("X0 shape before add lat lng ", X0.shape)
        X0 = self._add_lat_long_features(X0, addr_df_train)
        print("X0 shape after add lat lng ", X0.shape)

        # 1) generate the zip code dummy variables
        X_zips = self.zip_encoder.generate_training_zipcode_dummy_vars(X0)
        X1 = pd.concat((X0, X_zips), axis=1)

        # 2) Generate the features from the description blurbs
        # fit and generate the SVD features from the blurbs
        self.blurb_features.compute_training_tfidf_matrix(addr_df_train.blurb)
        tfidf_mat = self.blurb_features.get_training_tfidf_matrix()
        svd_df = self.blurb_features.compute_training_svd_df(training_tfidf_matrix=tfidf_mat)
        X2 = self._merge_svd_feats_and_apt_df(svd_df, X1, addr_df_train)

        # 3) Generate the cross-terms
        first_columns = list(X_zips.columns)
        second_columns = ['beds', 'baths', 'sq_ft', 'units_in_building']
        column_pairs = [('sq_ft', 'sq_ft'), ('beds', 'beds'), ('baths', 'baths')]
        self.cross_term_computer = CrossTermComputer(first_columns=first_columns, second_columns=second_columns, column_pairs=column_pairs)
        X_cross_terms = self.cross_term_computer.compute_cross_terms(X2)
        X3 = pd.concat((X2, X_cross_terms), axis=1)

        # 4) Generate bullet features (dummy vars and latent concepts)
        bullet_svd_df = self.bullet_features.get_training_svd_df(addr_df_train)
        X4 = self._merge_bullet_svd_and_apt_df(bullet_svd_df, X3, addr_df_train)

        # 5) Other features
        X5 = self._add_property_type(X4, addr_df_train)

        y = X5[['price']]
        X5 = X5.drop(columns=['price'])
        return (X5, y)
    
    def get_testing_features(self, apt_df_test, addr_df_test):
        X0 = apt_df_test.sort_values(by=['address']).reset_index(drop=True)

        X0 = self._add_lat_long_features(X0, addr_df_test)

        X_zips = self.zip_encoder.generate_testing_zipcode_dummy_vars(X0)
        X1 = pd.concat((X0, X_zips), axis = 1)

        svd_df = self.blurb_features.compute_testing_svd_df_from_blurbs(addr_df_test.blurb)
        X2 = self._merge_svd_feats_and_apt_df(svd_df, X1, addr_df_test)

        X_cross_terms = self.cross_term_computer.compute_cross_terms(X2)
        X3 = pd.concat((X2, X_cross_terms), axis=1)

        bullet_svd_df = self.bullet_features.get_testing_svd_df(addr_df_test)
        X4 = self._merge_bullet_svd_and_apt_df(bullet_svd_df, X3, addr_df_test)
        
        X5 = self._add_property_type(X4, addr_df_test)

        y = X5[['price']]
        X5 = X5.drop(columns=['price'])

        return (X5, y)



def select_column_names(all_column_names,
                        latlng = False,
                        units_in_building = False,
                        zip_feats = False,
                        blurb_feats = False,
                        bullet_feats = False,
                        building_type = False,
                        cross_term_res = []):
    returned_cols = ['beds', 'baths', 'sq_ft']  # we will always include these features
    if units_in_building:
        returned_cols.append('units_in_building')
    
    if latlng:
        latlng_cols = ['lat', 'lng', 'lat^2', 'lng^2', 'lat * lng']
        returned_cols.extend(latlng_cols)
    
    if zip_feats:
        zip_re = r'^zip_2[0-9]{4}$'
        for col in all_column_names:
            m = re.match(zip_re, col)
            if m:
                returned_cols.append(col)
    
    if blurb_feats:
        blurb_re = r'^svd_[0-9]+$'
        for col in all_column_names:
            m = re.match(blurb_re, col)
            if m:
                returned_cols.append(col)
    
    if bullet_feats:
        bullets_re = r'^Bullet_concept_.*'
        for col in all_column_names:
            m = re.match(bullets_re, col)
            if m:
                returned_cols.append(col)
    if building_type:
        type_cols = ['apartment', 'condo', 'house', 'townhouse', 'undefined']
        returned_cols.extend(type_cols)
    
    if cross_term_res is None:
        return returned_cols
        
    cross_term_cols = [col for col in all_column_names if col.find(' x ') != -1]
    returned_cross_term_cols = set()
    for col in cross_term_cols:
        for regexp in cross_term_res:
            m = re.match(regexp, col)
            if m:
                returned_cross_term_cols.add(col)
    returned_cols.extend(returned_cross_term_cols)
    return returned_cols

