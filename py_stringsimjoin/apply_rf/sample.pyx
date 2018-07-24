
import random

import pandas as pd

from py_stringsimjoin.utils.generic_helper import get_output_header_from_tables    

from cython.parallel import prange                                              

from libc.math cimport ceil                                                                                
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair                                                   
from libcpp.algorithm cimport sort 
                                                                                
from py_stringsimjoin.apply_rf.inverted_index cimport InvertedIndex             
from py_stringsimjoin.apply_rf.utils cimport build_inverted_index   
from py_stringsimjoin.apply_rf.tokenizers cimport tokenize_without_materializing

def sample_cython(ltable, rtable, l_key_attr, r_key_attr,                        
                 l_join_attr, r_join_attr, sample_size, y_param, seed, l_out_prefix='l_', r_out_prefix='r_'):
     
    cdef vector[pair[int, int]] sample
    cdef vector[string] lstrings, rstrings
    cdef vector[int] l_ids, r_ids
    convert_to_string_vector(ltable[l_join_attr], lstrings)
    convert_to_string_vector(rtable[r_join_attr], rstrings)                     
    convert_to_int_vector(ltable[l_key_attr], l_ids)
    convert_to_int_vector(rtable[r_key_attr], r_ids)                            
    
    sample_pairs(lstrings, rstrings, sample_size, y_param, sample)
 
    output_rows = []
    cdef pair[int, int] entry
    for entry in sample:
        output_rows.append([l_ids[entry.first], r_ids[entry.second]])
                     
    for seed_pair_row in seed.itertuples(index=False):                          
        output_rows.append([seed_pair_row[0], seed_pair_row[1]])                
                                                                                
    output_header = get_output_header_from_tables(l_key_attr, r_key_attr,       
                                                  None, None,                   
                                                  l_out_prefix, r_out_prefix)   
                                                                                
    output_table = pd.DataFrame(output_rows, columns=output_header)             
                                                                                
    # add an id column named '_id' to the output table.                         
    output_table.insert(0, '_id', range(0, len(output_table)))                  
                                                                                
    return output_table       

cdef void convert_to_string_vector(string_col, vector[string]& string_vector):        
    for val in string_col:                                                      
        string_vector.push_back(str(val))        

cdef void convert_to_int_vector(int_col, vector[int]& int_vector):  
    for val in int_col:                                                      
        int_vector.push_back(int(val))   

cdef void sample_pairs(vector[string]& lstrings, vector[string]& rstrings, 
                  int sample_size, int y_param, vector[pair[int, int]]& sample):
    cdef vector[vector[int]] ltokens, rtokens
    cdef string tok_type = "ws"

    # tokenize input strings using whitespace tokenizer
    tokenize_without_materializing(lstrings, rstrings, tok_type,            
                                   ltokens, rtokens)

    cdef int number_of_r_tuples_to_sample = <int>ceil(<float>sample_size / <float>y_param)
    sample_rtable_indices = random.sample(range(0, rstrings.size()),          
                                          number_of_r_tuples_to_sample)         
    cdef int cand_pos_ltuples_required = <int>ceil(y_param / 2.0)    

    # create inverted index over tokens in lstrings
    cdef InvertedIndex index                                                    
    build_inverted_index(ltokens, index)

    cdef oset[int] sample_ltable_indices
    cdef omap[int, int] candidate_overlap                                       
    cdef vector[int] candidates, tokens
    cdef vector[pair[int, int]] tmp
    cdef pair[int, int] entry
    cdef int i, j, m=0, k = 0, n=len(sample_rtable_indices)  
     
    for i in sample_rtable_indices:
        tokens = rtokens[i]                                                     
        print m, n
        m += 1                                                                            
        for j in range(tokens.size()):                                                      
            candidates = index.index[tokens[j]]                                 
            for cand in candidates:                                             
                candidate_overlap[cand] += 1 
        for entry in candidate_overlap:
            tmp.push_back(entry)

        sort(tmp.begin(), tmp.end(), comp) 

        k = 0
        for entry in tmp:
            sample_ltable_indices.insert(entry.first)
            k += 1
            if k == cand_pos_ltuples_required:
                break

        while sample_ltable_indices.size() < y_param:
            rand_idx = random.randint(0, lstrings.size())
            sample_ltable_indices.insert(rand_idx)

        for k in sample_ltable_indices:
            sample.push_back(pair[int, int](k, i))                        
        
        candidate_overlap.clear()
        sample_ltable_indices.clear()
        tmp.clear()

cdef bool comp(const pair[int, int]& l, const pair[int, int]& r):
    return l.second > r.second   
