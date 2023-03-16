import pandas as pd
import numpy as np 


import re



from sklearn.decomposition import TruncatedSVD

class _HelperMethods:
    @staticmethod
    def bullets_to_set(bullets_str):
        if bullets_str is None:
            return []
        
        bullets_str = str(bullets_str)

        bullets = bullets_str.split('()')[1:]

        discard_characters_re = r'[^\d\sa-z]'
        bullets = [b.strip().lower() for b in bullets]
        bullets = [re.sub(discard_characters_re, '', b) for b in bullets]
        return set(bullets)
    
    @staticmethod
    def to_1D(series):
        return pd.Series([x for _list in series for x in _list])

    @staticmethod
    def get_frequent_terms(df, min_count=10):
        term_to_count = _HelperMethods.to_1D(df.bullets_set).value_counts().to_dict()
        freq_terms = term_to_count
        terms_to_remove = []
        for t, c in freq_terms.items():
            if c < min_count:
                terms_to_remove.append(t)
        for t in terms_to_remove:
            freq_terms.pop(t)
        return freq_terms

    @staticmethod
    def remove_infrequent_terms(bullets_set, freq_terms):
        bullets_to_remove = set()
        for bullet in bullets_set:
            if bullet not in freq_terms:
                bullets_to_remove.add(bullet)
        return bullets_set - bullets_to_remove

    @staticmethod
    def generate_bullets_from_blurb(row, freq_terms):
        blurb = str(row.blurb).lower()
        terms = set()
        for term in freq_terms:
            if blurb.find(term) != -1:
                terms.add(term)
        fua = 'furnished units available'
        f = 'furnished'
        if fua in terms:
            terms.remove(f)
        terms.add('generated_from_blurb')
        return terms
    
    @staticmethod
    def combine_bullets_sets(row):
        bullets_set = row.bullets_set
        bullets_set_from_blurb = row.bullets_set_from_blurb
        combined_bullets = bullets_set.union(bullets_set_from_blurb)
        if len(bullets_set) == 0:
            combined_bullets.add("empty_bullets_set")
        return combined_bullets

    


class BulletFeatures:
    def __init__(self, min_bullet_count = 10, num_svd_components = 15):
        self.svd = None
        self.training_dummy_var_df = None
        self.min_bullet_count = min_bullet_count
        self.num_svd_components = num_svd_components
        self.training_bullet_to_count = None
    



    def _create_combined_bullets(self, addr_df, terms_to_freq = None):
        df = pd.DataFrame(addr_df[['address', 'blurb']])
        print("We made it here!")
        df['bullets_set'] = addr_df.bullets.map(_HelperMethods.bullets_to_set)
        print("But did we make it here?")
        if terms_to_freq is None:
            terms_to_freq = _HelperMethods.get_frequent_terms(df=df, min_count=self.min_bullet_count)
            self.training_bullet_to_count = terms_to_freq  # save this count to use at test time
        df['bullets_set'] = df.bullets_set.map(lambda bullets_set : _HelperMethods.remove_infrequent_terms(bullets_set, terms_to_freq))
        df['bullets_set_from_blurb'] = df.apply(lambda row: _HelperMethods.generate_bullets_from_blurb(row, freq_terms=terms_to_freq), axis=1)
        df['combined_bullets_set'] = df.apply(lambda row : _HelperMethods.combine_bullets_sets(row), axis=1)
        def bullets_set_to_string(bullets):
            return '|'.join(bullets)
        df['combined_bullets'] = df.combined_bullets_set.map(bullets_set_to_string)
        return df

    
    def generate_training_bullet_dummy_vars(self, addr_df_train):
        bullets_df = self._create_combined_bullets(addr_df_train)
        # add bullets to addr_df so that we can inspect manually
        addr_df_train['combined_bullets'] = bullets_df.combined_bullets
        dummy_var_df = bullets_df.combined_bullets.str.get_dummies(sep='|')
        self.training_dummy_var_df = dummy_var_df
        return dummy_var_df

    def get_training_svd_df(self, addr_df_train):
        dummy_var_df = self.generate_training_bullet_dummy_vars(addr_df_train)
        svd = TruncatedSVD(n_components=self.num_svd_components)
        self.svd = svd
        svd_matrix = svd.fit_transform(dummy_var_df)
        svd_df = pd.DataFrame(data=svd_matrix, columns=[f'Bullet_concept_{i}' for i in range(0, self.num_svd_components)])
        return svd_df

    def generate_testing_bullet_dummy_vars(self, addr_df_test):
        bullets_df = self._create_combined_bullets(addr_df_test, terms_to_freq = self.training_bullet_to_count)
        # add bullets to addr_df so that we can inspect manually
        addr_df_test['combined_bullets'] = bullets_df.combined_bullets

        dummy_var_df = bullets_df.combined_bullets.str.get_dummies(sep='|')
        dummy_cols = set(dummy_var_df.columns)
        training_dummy_cols = set(self.training_dummy_var_df.columns)
        missing_cols = list(training_dummy_cols - dummy_cols)
        for col in missing_cols:
            print("Adding dummy col " + col + "to test/validation bullet dummy df.")
            dummy_var_df[col] = 0
        return dummy_var_df
    
    def get_testing_svd_df(self, addr_df_test):
        dummy_var_df = self.generate_testing_bullet_dummy_vars(addr_df_test)
        svd_matrix = self.svd.transform(dummy_var_df)
        svd_df = pd.DataFrame(data=svd_matrix, columns=[f'Bullet_concept_{i}' for i in range(0, self.num_svd_components)])
        return svd_df 