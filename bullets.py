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
        return terms
    
    @staticmethod
    def combine_bullets_sets(row):
        bullets_set = row.bullets_set
        bullets_set_from_blurb = row.bullets_set_from_blurb
        combined_bullets = bullets_set.union(bullets_set_from_blurb)
        if len(bullets_set) == 0:
            combined_bullets.add("only_blurb_bullets")
        return combined_bullets

    


class BulletFeatures:
    def __init__(self, min_bullet_count = 10, num_svd_components = 15):
        self.svd = None
        self.training_dummy_var_df = None
        self.min_bullet_count = min_bullet_count
        self.num_svd_components = num_svd_components
        self.training_bullet_to_count = None
    



    def _create_combined_bullets(self, addr_df, training):
        ''' Uses the blurbs to create bullets for addresses missing them in the bullet list alone.
        '''
        print("Bryce?")
        print("here are the columns again in _create_combined_bullets: ", list(addr_df.columns))
        df = pd.DataFrame(addr_df[['address', 'blurb']])
        print("Richards?")
        df['bullets_set'] = addr_df.bullets.map(_HelperMethods.bullets_to_set)
        if training:
            terms_to_freq = _HelperMethods.get_frequent_terms(df=df, min_count=self.min_bullet_count)
            self.training_bullet_to_count = terms_to_freq  # save this count to use at test time
        elif training == False and self.training_bullet_to_count is None:
            raise ValueError("training param is false, but self.training_bullet_to_count has not been instantiated!")

        df['bullets_set'] = df.bullets_set.map(lambda bullets_set : _HelperMethods.remove_infrequent_terms(bullets_set, self.training_bullet_to_count))
        df['bullets_set_from_blurb'] = df.apply(lambda row: _HelperMethods.generate_bullets_from_blurb(row, freq_terms=self.training_bullet_to_count), axis=1)
        df['combined_bullets_set'] = df.apply(lambda row : _HelperMethods.combine_bullets_sets(row), axis=1)
        def bullets_set_to_string(bullets):
            return '|'.join(bullets)
        df['combined_bullets'] = df.combined_bullets_set.map(bullets_set_to_string)
        return df
    
    def generate_hand_picked_features(self, addr_df, training):
        bullets_df = self._create_combined_bullets(addr_df, training)
        # add this to the input df so that we can inspect manually
        # addr_df['combined_bullets'] = bullet_feats_df.combined_bullets

        # We'll construct some features by summing the contribution from multiple bullets.
        # These are represented by the dicts.
        # Other features will just be 0/1 - for instance, is the apartment furnished or not.
        KITCHEN_QUALITY_MAP = {'stainless steel appliances' : 1,
                       'granite countertops' : 2,
                       'breakfast nook' : 1,
                       'island kitchen': 1}
        SECURITY_TERMS = {'gated', 'controlled access'}
        LAWN_TERMS = {'lawn', 'yard'}
        PETS_ALLOWED_TERMS = {'dogs allowed', 'cats allowed', 'pet play area', 'pet washing station'}
        NO_PETS_ALLOWED_TERMS = {'no dogs allowed', 'no cats allowed', 'no pets allowed', 'pets not allowed'}
        UPSCALE_DETAILS_MAP = {'tile floors' : 1,
                            'hardwood floors' : 1,
                            'double pane windows' : 1,
                            'vaulted ceilings' : 1,
                            'crown molding' : 1,
                            'builtin bookshelves' : 1}
        POOL_TERMS = {'pool', 'swimming pool'}
        ELEVATOR_TERMS = {'elevator'}
        LUXURY_DETAILS_MAP = {'roof terrace' : 1,
                            'spa' : 1,
                            'concierge' : 1,
                            'onsite retail' : 1,
                            'penthouse' : 2,
                            'pet care' : 1}
        WATERFRONT_TERMS = {'waterfront', 'dock'}
        RECREATION_DETAILS_MAP = {'fitness center' : 1,
                                'tennis courts' : 1,
                                'volleyball court' : 1,
                                'basketball court' : 0.5}
        FURNISHED_TERMS = {'furnished', 'fully furnished'}
        BULLET_FEAT_PREFEX = 'bullets_feat_'
        FEATURE_TO_TERMS_MAP = {
            'has_security' : SECURITY_TERMS,
            'has_lawn' : LAWN_TERMS,
            'are_pets_allowed' : PETS_ALLOWED_TERMS,
            'are_no_pets_allowed' : NO_PETS_ALLOWED_TERMS,
            'has_pool' : POOL_TERMS,
            'is_waterfront' : WATERFRONT_TERMS,
            'is_furnished' : FURNISHED_TERMS,
            'has_elevator' : ELEVATOR_TERMS,
            'kitchen_quality_score' : KITCHEN_QUALITY_MAP,
            'upscale_score' : UPSCALE_DETAILS_MAP,
            'recreation_score' : RECREATION_DETAILS_MAP,
            'luxury_score' : LUXURY_DETAILS_MAP
        }
        # Helper methods to handle to score/bool cases.
        def __compute_score(bullets_set, term_to_score_map):
            score = 0.0
            for bullet in bullets_set:
                if bullet in term_to_score_map:
                    score += term_to_score_map[bullet]
            return score 

        def __compute_bool(bullets_set, terms_set):
            for bullet in bullets_set:
                if bullet in terms_set:
                    return 1
            return 0
        
        # Finally we're ready to use all the above code. Loop through each feature and call
        # __compute_score() or __compute_bool()
        bullet_feats_df = pd.DataFrame()
        for feat_name, terms in FEATURE_TO_TERMS_MAP.items():
            if feat_name.endswith('_score'):
                bullet_feats_df[BULLET_FEAT_PREFEX + feat_name] = bullets_df.combined_bullets_set.map(lambda bullets_set : __compute_score(bullets_set, terms))
            else:
                bullet_feats_df[BULLET_FEAT_PREFEX + feat_name] = bullets_df.combined_bullets_set.map(lambda bullets_set : __compute_bool(bullets_set, terms))
        
        return bullet_feats_df
    
    def generate_training_bullet_dummy_vars(self, addr_df_train):
        bullets_df = self._create_combined_bullets(addr_df_train, training=True)
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
        bullets_df = self._create_combined_bullets(addr_df_test, training=False)
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