import copy
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.activations import linear, relu, sigmoid

def calculate_diff_from_median_price(apt_df, addr_df, apt_control_vars=['beds'], addr_control_vars=['zip']):
    ''' Per-address, calculate how far away from the median rent that address's apartments are.

    control_vars determines how the median is calculated. For instance, if addr_control_vars = ['zips'],
    then the median rent per zip code is calculated, cal this med_zips.

    For the j'th apartment in an address in zip code i, let price_diffs[j] = price[j] - med_zips[i]

    For this address, avg_price_diffs = mean(price_diffs). This number captures how expensive an apartment is compared to similar apartments (those sharing 
    the same control_vars).

    returns a dataframe with columns 'address' and 'avg_diff_from_median'.
    '''
    addr_df_copy = copy.copy(addr_df)
    addr_columns = addr_control_vars + ['address']
    addr_df_copy = addr_df_copy[addr_columns]

    apt_df_copy = copy.copy(apt_df)
    apt_columns = apt_control_vars + ['address', 'price']
    apt_df_copy = apt_df_copy[apt_columns]

    df = apt_df_copy.merge(addr_df_copy, on='address')
    control_vars = apt_control_vars + addr_control_vars
    median_rent = df.groupby(control_vars).median(numeric_only=True)
    median_rent.rename(columns={'price' : 'median_price'}, inplace=True)

    combined = median_rent.merge(df, on=['beds', 'zip'])
    assert(combined.shape[0] == df.shape[0])
    combined['price_diff'] = combined.price - combined.median_price
    avg_price_diff = combined.groupby('address')['price_diff'].mean().to_frame()
    ret_df = avg_price_diff.rename(columns={'price_diff' : 'avg_diff_from_median'})
    return ret_df

def count_vectorizer_output_to_dataframe(counts, count_vectorizer):
    pass


# Pick params systematically
def search_hyperparams(
    X_train, y_train, X_test, y_test, epochs = 1500,
    node_layer_architectures = [(5), (10), (20), (40), (10, 5), (20, 10), (40, 20), (10, 10, 10), (20, 10, 5)],
    lambdas = [0.01, .05, 0.1, 0.2, 0.5, 1.0]):
    model_name_to_model = dict()
    model_name_to_history = dict()
    best_testing_result = 9999999
    best_model = ""
    normalizer = Normalization(axis=-1)
    normalizer.adapt(X_train)
    for lambda_ in lambdas:
        for node_sequence in node_layer_architectures:
            # Here we'll construct the layers of our neural net
            layers = []
            # Add all the hidden layers
            for i in range(len(node_sequence)):
                n_nodes = node_sequence[i]
                name = "Layer_" + str(i)
                layer = Dense(n_nodes, activation='relu', name=name, kernel_regularizer=tf.keras.regularizers.L2(lambda_))
                layers.append(layer)
            # Add the output layer
            layers.append(Dense(1, activation='linear'))
            model_name = "hidden-layers( {nodes} )_lambda( {lambda_} )".format(
                nodes=str(node_sequence), lambda_=str(lambda_))
            model = model = tf.keras.models.Sequential(
                normalizer=normalizer,
                layers=layers, 
                name=model_name)
            print("About to train model: ", model_name)
            history = model.fit(X_train, y_train, epochs=epochs, verbose=0)

            model_name_to_history[model_name] = history 
            model_name_to_model[model_name] = model
            results_train = model.evaluate(X_train, y_train, verbose=0)
            results_test = model.evaluate(X_test, y_test, verbose=0)
            print("Model ", model_name, "'s performance:")
            print("Training: ", results_train, ". Testing: ", results_test)
            if results_test < best_testing_result:
                best_testing_result = results_test
                best_model = model_name
                
    print("The best model was " + best_model + " with testing result: " + best_testing_result)
    return (model_name_to_model, model_name_to_history)
            