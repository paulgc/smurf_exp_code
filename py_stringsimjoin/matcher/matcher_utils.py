
import math
from collections import OrderedDict

import pandas as pd
import sklearn.cross_validation as cv

def split_train_test(labeled_data, target_attr, train_proportion=0.5,
                     random_state=None, verbose=True):
    """
    This function splits the input data into train and test.

    Specifically, this function is just a wrapper of scikit-learn's
    train_test_split function.

    This function also takes care of copying the metadata from the input
    table to train and test splits.

    Args:
        labeled_data (DataFrame): The input pandas DataFrame that needs to be
            split into train and test.
        train_proportion (float): A number between 0 and 1, indicating the
            proportion of tuples that should be included in the train split (
            defaults to 0.5).
        random_state (object): A number of random number object (as in
            scikit-learn).
        verbose (boolean): A flag to indicate whether the debug information
            should be displayed.

    Returns:

        A Python dictionary containing two keys - train and test.

        The value for the key 'train' is a pandas DataFrame containing tuples
        allocated from the input table based on train_proportion.

        Similarly, the value for the key 'test' is a pandas DataFrame containing
        tuples for evaluation.

        This function sets the output DataFrames (train, test) properties
        same as the input DataFrame.

    """
    #  We expect labeled data to be of type pandas DataFrame
    if not isinstance(labeled_data, pd.DataFrame):
        raise AssertionError('Input table is not of type DataFrame')

    num_rows = len(labeled_data)

    # We expect the train proportion to be between 0 and 1.
    assert train_proportion >= 0 and train_proportion <= 1, \
        " Train proportion is expected to be between 0 and 1"

    # We expect the number of rows in the table to be non-empty
    assert num_rows > 0, 'The input table is empty'

    # Explicitly get the train and test size in terms of tuples (based on the
    #  given proportion)
    train_size = int(math.floor(num_rows * train_proportion))
    test_size = int(num_rows - train_size)

    # Use sk-learn to split the data
    idx_values = pd.np.array(labeled_data.index.values)
    idx_train, idx_test = cv.train_test_split(idx_values, test_size=test_size,
                                              train_size=train_size,
                                              random_state=random_state,
                                              stratify=labeled_data[target_attr].values)

    # Construct output tables.
    labeled_train = labeled_data.ix[idx_train]
    labeled_test = labeled_data.ix[idx_test]

    # Return output tables
    result = OrderedDict()
    result['train'] = labeled_train
    result['test'] = labeled_test

    # Finally, return the dictionary.
    return result

def get_threshold_from_matcher(matcher):
#    intercept = matcher.intercept_
#    intercept *= -1
    return  - float(matcher.intercept_)/float(matcher.coef_)
