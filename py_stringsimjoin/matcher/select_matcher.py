from collections import OrderedDict

import pandas as pd
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from py_stringsimjoin.matcher.matcher_utils import get_threshold_from_matcher

def select_matcher(feature_vectors, target_attr, feature_table,
                   metric='precision', k=5,
                   random_state=None):
    """
    This function selects a matcher from a given list of matchers based on a
    given metric.
    
    Specifically, this function internally uses scikit-learn's
    cross validation function to select a matcher. There are two ways the
    user can call the fit method. First, interface similar to scikit-learn
    where the feature vectors and target attribute given as projected DataFrame.
    Second, give the DataFrame and explicitly specify the feature vectors
    (by specifying the attributes to be excluded) and the target attribute

    A point to note is all the input parameters have a default value of
    None. This is done to support both the interfaces in a single function.

    Args:
        matchers (MLMatcher): List of ML matchers to be selected from.
        x (DataFrame): Input feature vectors given as pandas DataFrame (
            defaults to None).
        y (DatFrame): Input target attribute given as pandas
            DataFrame with a single column (defaults to None).
        table (DataFrame): Input pandas DataFrame containing feature
            vectors and target attribute (defaults to None).
        exclude_attrs (list): The list of attributes that should be
            excluded from the input table to get the feature vectors.
        target_attr (string): The target attribute in the input table (defaults
            to None).
        metric (string): The metric based on which the matchers must be
            selected. The string can be one of 'precision', 'recall',
            'f1' (defaults to 'precision').
        k (int): The k value for cross-validation (defaults to 5).
        random_state (object): Pseudo random number generator that should be
            used for splitting the data into folds (defaults to None).

    Returns:
        A dictionary containing two keys - selected matcher and the cv_stats.

        The selected matcher has a value that is a matcher (MLMatcher) object
        and cv_stats has a value that is a dictionary containing
        cross-validation statistics.

    """
    fv_table_col_names = list(feature_vectors.columns)
    feature_col_names = []
    for col_name in fv_table_col_names:
        if col_name in feature_table.index:
            feature_col_names.append(col_name)
    matchers = {}
    for feat_name in feature_col_names:
        matchers[feat_name] = LogisticRegression()

    dict_list = []
    max_score = 0
    # Initialize the best matcher. As of now set it to be the first matcher.
    sel_matcher = matchers[matchers.keys()[0]]

    # Fix the header
    header = ['Name', 'Num folds']

    # Append the folds
    fold_header = ['Fold ' + str(i + 1) for i in range(k)]
    header.extend(fold_header)

    # Finally, append the score.
    header.append('Mean score')

    cv = KFold(len(feature_vectors), k, shuffle=True, random_state=random_state)              

    for matcher_name in matchers.keys():
        # Use scikit learn's cross validation to get the matcher and the list
        #  of scores (one for each fold).
        scores = cross_val_score(matchers[matcher_name], 
                                 feature_vectors[[matcher_name]].values, 
                                 feature_vectors[target_attr].values, 
                                 metric, cv)

        # Fill a dictionary based on the matcher and the scores.
        val_list = [matcher_name, k]
        val_list.extend(scores)
        val_list.append(pd.np.mean(scores))
        d = OrderedDict(zip(header, val_list))
        dict_list.append(d)

        # Select the matcher based on the mean scoere.
        if pd.np.mean(scores) > max_score:
            sel_matcher = matchers[matcher_name]
            max_score = pd.np.mean(scores)

    # Create a DataFrame based on the list of dictionaries created
    stats = pd.DataFrame(dict_list)
    stats = stats[header]
    res = OrderedDict()
    # Add selected matcher and the stats to a dictionary.
    res['selected_matcher'] = sel_matcher
    res['cv_stats'] = stats

    # Return the final dictionary containing selected matcher and the CV
    # statistics.
    return res


def cross_validation(matcher, x, y, metric, k, random_state):
    """
    The function does cross validation for a single matcher
    """
    # Use KFold function from scikit learn to create a cv object that can be
    # used for cross_val_score function.
    cv = KFold(len(y), k, shuffle=True, random_state=random_state)
    # Call the scikit-learn's cross_val_score function
    scores = cross_val_score(matcher, x, y, scoring=metric, cv=cv)
    # Finally, return the matcher along with the scores.
    return matcher, scores

