"""Similarity measure utilities"""

from py_stringmatching.similarity_measure.cosine import Cosine
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.overlap_coefficient import \
                                                            OverlapCoefficient


def get_sim_function(sim_measure_type):
    if sim_measure_type == 'COSINE':
        return Cosine().get_raw_score
    elif sim_measure_type == 'DICE':
        return Dice().get_raw_score
    elif sim_measure_type == 'EDIT_DISTANCE':
        return Levenshtein().get_raw_score
    elif sim_measure_type == 'JACCARD':
        return Jaccard().get_raw_score
    elif sim_measure_type == 'OVERLAP':
        return overlap
    elif sim_measure_type == 'OVERLAP_COEFFICIENT':
        return OverlapCoefficient().get_raw_score
    elif sim_measure_type == 'LEFT_LENGTH':
        return left_length
    elif sim_measure_type == 'RIGHT_LENGTH':
        return right_length
    elif sim_measure_type == 'LENGTH_SUM':
        return length_sum
    elif sim_measure_type == 'LENGTH_DIFF':
        return length_diff

def left_length(str1, str2):
    return len(str1)

def right_length(str1, str2):                                                    
    return len(str2)

def length_sum(str1, str2):                                                    
    return len(str1) + len(str2)  

def length_diff(str1, str2):                                                     
    return abs(len(str1) - len(str2))

def overlap(set1, set2):
    """Computes the overlap between two sets.

    Args:
        set1,set2 (set or list): Input sets (or lists). Input lists are 
            converted to sets.

    Returns:
        overlap (int)
    """
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    return len(set1.intersection(set2))
