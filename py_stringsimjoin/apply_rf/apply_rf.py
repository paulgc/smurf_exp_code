
from six import iteritems                                                       
import pandas as pd                                                             
                                                                                
from py_stringsimjoin.apply_rf.extract_rules import extract_pos_rules_from_tree   
from py_stringsimjoin.utils.converter import dataframe_column_to_str

def merge_candsets(candset_list, candset_l_key_attr, candset_r_key_attr, num_trees,        
                   vote_col='votes'):                                           
    print len(candset_list), candset_l_key_attr, candset_r_key_attr
    vote_cnt = {}                                                               
    for candset in candset_list:                                                
        # Find indices of l_key_attr and r_key_attr in candset                      
        candset_columns = list(candset.columns.values)                          
        candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)    
        candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)    
        dataframe_column_to_str(candset, candset_l_key_attr, inplace=True)
        dataframe_column_to_str(candset, candset_r_key_attr, inplace=True)      
        for candset_row in candset.itertuples(index=False):                     
            pair_id = str(candset_row[candset_l_key_attr_index])+','+str(candset_row[candset_r_key_attr_index])
            curr_votes = vote_cnt.get(pair_id, 0)                               
            vote_cnt[pair_id] = curr_votes + 1                                  
    output_rows = []                                                            
    for pair_id, votes in iteritems(vote_cnt):                            
        if votes >= (num_trees/2.0):      
            fields = pair_id.split(',')                                             
            output_rows.append([fields[0], fields[1], votes])                       
    return pd.DataFrame(output_rows, columns=['l_id', 'r_id', vote_col])        
                                                                                
def apply_rf(ltable, rtable, l_key_attr, r_key_attr,                            
             l_match_attr, r_match_attr, rf, feature_table, n_jobs=1):          
    rule_sets = []                                                              
    for dt in rf.estimators_:                                                   
        rule_sets.append(extract_pos_rules_from_tree(dt, feature_table))                      

    return apply_rulesets(ltable, rtable, l_key_attr, r_key_attr,                            
                          l_match_attr, r_match_attr, rule_sets, n_jobs=1)

def apply_rulesets(ltable, rtable, l_key_attr, r_key_attr,                            
                   l_match_attr, r_match_attr, rule_sets, n_jobs=1):                   
    rule_set_outputs = []                                                       
    for rule_set in rule_sets:                                                  
        rule_set_outputs.append(rule_set.apply_tables(ltable, rtable,           
                                                  l_key_attr, r_key_attr,       
                                                  l_match_attr, r_match_attr,   
                                                  n_jobs))                      
    return merge_candsets(rule_set_outputs, 'l_'+l_key_attr, 'r_'+r_key_attr, len(rule_sets))  
