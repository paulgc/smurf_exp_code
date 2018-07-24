
from math import ceil
import operator                              
import random

from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
import pandas as pd
import pyprind                                                                  

from py_stringsimjoin.filter.overlap_filter import OverlapFilter
from py_stringsimjoin.index.inverted_index import InvertedIndex                 
from py_stringsimjoin.utils.generic_helper import convert_dataframe_to_array, \
    get_attrs_to_project, get_output_header_from_tables


def sample_pairs(ltable, rtable, l_key_attr, r_key_attr, 
                 l_join_attr, r_join_attr, sample_size, y_param, seed,
                 l_out_prefix='l_', r_out_prefix='r_', show_progress=True):
    # get attributes to project.                                                
    l_proj_attrs = get_attrs_to_project(None, l_key_attr, l_join_attr)   
    r_proj_attrs = get_attrs_to_project(None, r_key_attr, r_join_attr)  

    # convert dataframe to array for faster access       
    ltable_array = convert_dataframe_to_array(ltable, l_proj_attrs, l_join_attr)
    rtable_array = convert_dataframe_to_array(rtable, r_proj_attrs, r_join_attr)

    # find column indices of key attr and join attr in ltable array                  
    l_key_attr_index = l_proj_attrs.index(l_key_attr)                              
    l_join_attr_index = l_proj_attrs.index(l_join_attr)                            
                                                                                
    # find column indices of key attr and join attr in rtable array                   
    r_key_attr_index = r_proj_attrs.index(r_key_attr)                              
    r_join_attr_index = r_proj_attrs.index(r_join_attr)  

    # create a whitespace tokenizer to tokenize join attributes                 
    ws_tok = WhitespaceTokenizer(return_set=True)     

    # build inverted index on join attriubute in ltable
    inverted_index = InvertedIndex(ltable_array, l_join_attr_index, ws_tok)
    inverted_index.build()

    number_of_r_tuples_to_sample = int(ceil(float(sample_size) / float(y_param)))   
    sample_rtable_indices = random.sample(range(0, len(rtable_array)),
                                          number_of_r_tuples_to_sample)
    cand_pos_ltuples_required = int(ceil(y_param / 2.0))                    

    overlap_filter = OverlapFilter(ws_tok, 1)                                

    output_rows = [] 

    if show_progress:                                                           
        prog_bar = pyprind.ProgBar(number_of_r_tuples_to_sample)    

    for r_idx in sample_rtable_indices:
        r_row = rtable_array[r_idx]
        r_id = r_row[r_key_attr_index]
        r_join_attr_tokens = ws_tok.tokenize(r_row[r_join_attr_index])

        # probe inverted index and find ltable candidates                   
        cand_overlap = overlap_filter.find_candidates(                     
                           r_join_attr_tokens, inverted_index)          

        sampled_ltuples = set() 
        for cand in sorted(cand_overlap.items(), key=operator.itemgetter(1), 
                           reverse=True):
            if len(sampled_ltuples) == cand_pos_ltuples_required:
                break 
            sampled_ltuples.add(cand[0])

        ltable_size = len(ltable_array)
        while len(sampled_ltuples) < y_param:
            rand_idx = random.randint(0, ltable_size - 1)
            sampled_ltuples.add(rand_idx)

        for l_idx in sampled_ltuples:
            output_rows.append([ltable_array[l_idx][l_key_attr_index], r_id])

        if show_progress:                                                       
            prog_bar.update()

    for seed_pair_row in seed.itertuples(index=False):                          
        output_rows.append([seed_pair_row[0], seed_pair_row[1]])
   
    output_header = get_output_header_from_tables(l_key_attr, r_key_attr,       
                                                  None, None,     
                                                  l_out_prefix, r_out_prefix)

    output_table = pd.DataFrame(output_rows, columns=output_header)
             
    # add an id column named '_id' to the output table.                         
    output_table.insert(0, '_id', range(0, len(output_table)))    

    return output_table           


def _get_stop_words():
    stop_words_set = set()
    stop_words_file = '/scratch/stop_words.txt'
    with open(stop_words_file, "rb") as stopwords_file:
        for stop_words in stopwords_file:
            stop_words_set.add(stop_words.rstrip())

    return stop_words_set 
