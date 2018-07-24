
from py_stringmatching.tokenizer.alphabetic_tokenizer import AlphabeticTokenizer
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
import pandas as pd 

from py_stringsimjoin.utils.simfunctions import get_sim_function
from py_stringsimjoin.utils.tokenizers import NumericTokenizer                

def get_features(sim_measures=None, tokenizers=None):
    features = []
    ws_tok = WhitespaceTokenizer(return_set=True)
    if sim_measures is None:
        sim_measures = ['JACCARD', 'COSINE', 'DICE', 
#                        'LEFT_LENGTH', 'RIGHT_LENGTH', 'LENGTH_SUM', 'LENGTH_DIFF']
                       'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE', 'LEFT_LENGTH', 'RIGHT_LENGTH', 'LENGTH_SUM', 'LENGTH_DIFF']
    if tokenizers is None:
        tokenizers = {'alph': AlphabeticTokenizer(return_set=True),
                      'alph_num': AlphanumericTokenizer(return_set=True),
                      'num': NumericTokenizer(return_set=True),
                      'ws': WhitespaceTokenizer(return_set=True),
                      'qg2': QgramTokenizer(qval=2, return_set=True),
                      'qg3': QgramTokenizer(qval=3, return_set=True)}
    for sim_measure_type in sim_measures:
        if sim_measure_type in ['EDIT_DISTANCE', 'LEFT_LENGTH', 'RIGHT_LENGTH', 'LENGTH_SUM', 'LENGTH_DIFF']:
            features.append((sim_measure_type.lower(), 'none', sim_measure_type,
                             None, get_sim_function(sim_measure_type)))
            continue
        for tok_name in tokenizers.keys():
#            if sim_measure_type == 'COSINE' and tok_name == 'qg3':
#                continue
            features.append((sim_measure_type.lower()+'_'+tok_name, tok_name, 
                             sim_measure_type, tokenizers[tok_name], 
                             get_sim_function(sim_measure_type)))

    feature_table_header = ['feature_name', 'tokenizer_type', 'sim_measure_type', 
                            'tokenizer', 'sim_function']
    feature_table = pd.DataFrame(features, columns=feature_table_header)
    feature_table = feature_table.set_index('feature_name')

    return feature_table     
