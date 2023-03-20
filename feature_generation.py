import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

import re

from copy import deepcopy

import imp
import bullets
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


class DummyEncoder:
    def __init__(self, col_name, min_count = 10):
        self.col_name = col_name
        self.min_count = min_count
        self.training_dummies_df = None
        self.encoder = None
        self.col_names = None

    # TODO: make this a general dummy_encoding_to_df utility method.
    def _convert_results_to_df(self, dummies_np):
        col_names = self.encoder.get_feature_names_out()
        col_names = [name.replace('x0', self.col_name) for name in col_names]
        infrequent_col_names = [name for name in col_names if name.find('infrequent') != -1]
        dummies_df = pd.DataFrame(dummies_np, columns=col_names)
        dummies_df.drop(columns=infrequent_col_names, inplace=True)
        return dummies_df
        
    def generate_training_dummy_vars(self, df, min_zipcode_count=10):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output = False, min_frequency=min_zipcode_count)
        dummies_np = self.encoder.fit_transform(df[[self.col_name]])  # note that here we do fit_transform
        dummies_df = self._convert_results_to_df(dummies_np)
        self.training_dummies_df = dummies_df
        return dummies_df
   
    def generate_testing_dummy_vars(self, df):
        n = df.shape[0]
        dummies_np = self.encoder.transform(df[[self.col_name]])  # note that here we just transform - encoder was fit on training data
        dummies_df = self._convert_results_to_df(dummies_np)
        return dummies_df



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


    # define indices for training and testing data
    num_addrs = addr_df.shape[0]
    addr_df = shuffle(addr_df).reset_index(drop=True)
    num_train = int(0.70 * num_addrs)
    num_test = num_addrs - num_train
    train_indices = pd.Series([True for i in range(num_train)] + [False for i in range(num_addrs - num_train)])
    test_indices = pd.Series([False for i in range(num_train)] + [True for i in range(num_test)])
    
    # create training and testing dataframes
    addr_df_train = addr_df.loc[train_indices].reset_index(drop=True)
    addr_df_test = addr_df.loc[test_indices].reset_index(drop=True)
    apt_df_train = pd.merge(apt_df, addr_df_train[['address']], on='address')
    apt_df_test = pd.merge(apt_df, addr_df_test[['address']], on='address')

    return ((apt_df_train, addr_df_train), (apt_df_test, addr_df_test))


class FeatureGenerator:
    def __init__(self, min_bullet_count = 20, num_svd_components = 15, min_zipcode_dummy_count = 10):
        self.cross_term_computer = None
        self.blurb_features = BlurbFeatures()
        
        # Dictionary mapping from column name to the DummyEncoder object for that column.
        # This will actually let us create new encoders on the fly in get_training_features().
        self.dummy_encoders = dict()
        self.dummy_encoders["zip"] = DummyEncoder(col_name="zip", min_count = min_zipcode_dummy_count)
        self.dummy_encoders["property_type"] = DummyEncoder(col_name="property_type", min_count = 3)
        self.property_type_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output = False, min_frequency=10)
        self.bullet_features = BulletFeatures(min_bullet_count=min_bullet_count,
                                              num_svd_components=num_svd_components)
    
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
        return (X, y)

    def _add_lat_long_features(self, addr_feats):
        addr_feats['lat'] = addr_feats.latlng.map(lambda s : self._extract_lat_and_lng(s)[0])
        addr_feats['lng'] = addr_feats.latlng.map(lambda s : self._extract_lat_and_lng(s)[1])
        addr_feats['lat^2'] = addr_feats.lat ** 2
        addr_feats['lng^2'] = addr_feats.lng ** 2
        addr_feats['lat * lng'] = addr_feats.lat * addr_feats.lng
        return addr_feats
    
    def _convert_ptype_to_df(self, ptype_dummies, encoder):
        col_names = encoder.get_feature_names_out()
        col_names = [name.replace('x0', 'ptype') for name in col_names]
        infrequent_col_names = [name for name in col_names if name.find('infrequent') != -1]
        ptype_df = pd.DataFrame(ptype_dummies, columns=col_names)
        ptype_df.drop(columns=infrequent_col_names, inplace=True)
        return ptype_df
    
    def _add_property_type(self, addr_feats):
        def type_from_title(title):
            m = re.match(r'.* ([A-Za-z]+) for Rent.*', title)
            if m is None:
                return 'undefined'
            type = m.group(1).lower()
            if type == 'houses':
                type = 'house'
            return type
        addr_feats['property_type'] = addr_feats.title.map(type_from_title)
        return addr_feats

    
    def _merge_bullet_svd_and_apt_df(self, bullet_svd_df, apt_df, addr_df):
        addrs = addr_df[['address']]
        addrs_and_bullet_svd_feats = addrs.join(bullet_svd_df)
        assert(addrs_and_bullet_svd_feats.shape[0] == bullet_svd_df.shape[0])
        X_with_bullets = apt_df.merge(addrs_and_bullet_svd_feats, on='address')
        return X_with_bullets
    
    def _add_bullet_feats(self, addr_df, addr_feats, training = True):
        bullet_svd_df = None
        if training:
            bullet_svd_df = self.bullet_features.get_training_svd_df(addr_df)
        else:
            bullet_svd_df = self.bullet_features.get_testing_svd_df(addr_df)
        addr_feats_new = addr_feats.join(bullet_svd_df)
        assert(addr_feats.shape == addr_feats.shape)
        return addr_feats_new
    
    def _add_blurb_feats(self, addr_feats, training = True):
        svd_df = None
        if training:
            self.blurb_features.compute_training_tfidf_matrix(addr_feats.blurb)
            tfidf_mat = self.blurb_features.get_training_tfidf_matrix()
            svd_df = self.blurb_features.compute_training_svd_df(training_tfidf_matrix=tfidf_mat)
        else:
            svd_df = self.blurb_features.compute_testing_svd_df_from_blurbs(addr_feats.blurb)
        addr_feats_new = pd.concat((addr_feats, svd_df), axis=1)
        return addr_feats_new
        

    def _add_furnished(self, addr_feats):
        def is_furnished(bullets):
            return int("()Furnished." in bullets)
        addr_feats['furnished'] = addr_feats.bullets.map(is_furnished)
        return addr_feats
    
    def _add_dummies(self, addr_feats, col_name, training=True):
        dummies_df = None
        if training:
            dummies_df = self.dummy_encoders[col_name].generate_training_dummy_vars(addr_feats)
        else:
            dummies_df = self.dummy_encoders[col_name].generate_testing_dummy_vars(addr_feats)
        addr_feats_new = pd.concat((addr_feats, dummies_df), axis = 1)
        return addr_feats_new
    
    def _merge_addr_and_apt_features(self, addr_features, apt_features):
        merged_features = apt_features.merge(addr_features, on='address')
        return merged_features

    def _convert_col_to_dummies(self, addr_features, col_name, training=True):
        """ stores encoder in self.col_to_encoders[col_name] """

    def _get_features(self, apt_df, addr_df, training):
        apt_features = apt_df.sort_values(by=['address']).reset_index(drop=True)
        addr_features = deepcopy(addr_df)
        addr_features = self._add_lat_long_features(addr_features)
        addr_features = self._add_blurb_feats(addr_features, training = training)
        addr_features = self._add_bullet_feats(addr_df=addr_df, addr_feats=addr_features, training=training)
        addr_features = self._add_furnished(addr_features)
        addr_features = self._add_dummies(addr_features, col_name="zip", training=training)
        # To add the property type dummy vars, first we need to create a "property_type" column
        # Perhaps this should be done in a separate step.
        addr_features = self._add_property_type(addr_features)
        addr_features = self._add_dummies(addr_features, col_name="property_type", training=training)

        X = self._merge_addr_and_apt_features(addr_features, apt_features)

        # Generate the cross-terms
        columns = list(addr_features.columns)
        zipcode_dummy_columns = [col for col in columns if 'zip_' in col]
        basic_columns = ['beds', 'baths', 'sq_ft', 'units_in_building']
        column_pairs = [('sq_ft', 'sq_ft'), ('beds', 'beds'), ('baths', 'baths')]
        self.cross_term_computer = CrossTermComputer(first_columns=zipcode_dummy_columns, second_columns=basic_columns, column_pairs=column_pairs)
        X_cross_terms = self.cross_term_computer.compute_cross_terms(X)
        X = pd.concat((X, X_cross_terms), axis=1)

        y = X[['price']]
        X = X.drop(columns=['price'])
        return (X, y)

    def get_training_features(self, apt_df_train, addr_df_train):
        (X, y) = self._get_features(apt_df_train, addr_df_train, training = True)
        return (X, y)
    
    def get_testing_features(self, apt_df_test, addr_df_test):
        (X, y) = self._get_features(apt_df_test, addr_df_test, training = False)
        return (X, y)


def select_column_names(all_column_names,
                        latlng = False,
                        units_in_building = False,
                        zip_feats = False,
                        blurb_feats = False,
                        bullet_feats = False,
                        property_type = False,
                        furnished = False,
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
    if property_type:
        # property_types = ['apartment', 'condo', 'house', 'townhouse', 'undefined']
        property_types = ['house', 'townhouse', 'undefined']
        type_cols = ['property_type_' + p_type for p_type in property_types]
        returned_cols.extend(type_cols)
    
    if furnished:
        returned_cols.append('furnished')
    
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

