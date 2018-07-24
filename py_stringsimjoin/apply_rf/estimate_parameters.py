
import time

import pandas as pd                                                             
import pyprind                                                                  
                                                                                
from py_stringsimjoin.utils.generic_helper import build_dict_from_table


def compute_coverage(rule_sets, fvs):
    for rule_set in rule_sets:
        ruleset_cov = None
        first_rule = True
        for rule in rule_set.rules:
            rule_cov = None                                                         
            first_pred = True             
            for predicate in rule.predicates:
                pred_cov = fvs[predicate.feat_name].apply(lambda f: 
                               predicate.comp_fn(f, predicate.threshold), 1)
                predicate.set_coverage(pred_cov)
                if first_pred:
                    rule_cov = pred_cov
                    first_pred = False
                else:
                    rule_cov = rule_cov & pred_cov
            rule.set_coverage(rule_cov)
            if first_rule:
                ruleset_cov = rule_cov
                first_rule = False
            else:
                ruleset_cov = ruleset_cov | rule_cov
        rule_set.set_coverage(ruleset_cov)

def compute_feature_costs(candset, candset_l_key_attr, candset_r_key_attr,       
                          ltable, rtable,                                        
                          l_key_attr, r_key_attr, l_join_attr, r_join_attr,      
                          feature_table, show_progress=True):
    # Find column indices of key attr and join attr in ltable                   
    l_columns = list(ltable.columns.values)                                     
    l_key_attr_index = l_columns.index(l_key_attr)                              
    l_join_attr_index = l_columns.index(l_join_attr)                            
                                                                                
    # Find column indices of key attr and join attr in rtable                   
    r_columns = list(rtable.columns.values)                                     
    r_key_attr_index = r_columns.index(r_key_attr)                              
    r_join_attr_index = r_columns.index(r_join_attr)                            
                                                                                
    # Find indices of l_key_attr and r_key_attr in candset                      
    candset_columns = list(candset.columns.values)                              
    candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)        
    candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)        
                                                                                
    # Build a dictionary on ltable                                              
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index,               
                                        l_join_attr_index,                      
                                        remove_null=False)                      
                                                                                
    # Build a dictionary on rtable                                              
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index,               
                                        r_join_attr_index,                      
                                        remove_null=False)

    feature_costs = {}
    for feature_name in feature_table.index:
        feature_costs[feature_name] = 0.0                   
         
    if show_progress:                                                           
        prog_bar = pyprind.ProgBar(len(candset))                                

    cnt = 0                                                                                
    for candset_row in candset.itertuples(index=False):                         
        if cnt == 10000:
            break
        cnt += 1
        l_id = candset_row[candset_l_key_attr_index]                            
        r_id = candset_row[candset_r_key_attr_index]                            
                                                                                
        l_string = ltable_dict[l_id][l_join_attr_index]                         
        r_string = rtable_dict[r_id][r_join_attr_index]                         
                                                                                
                                                                                
        # compute feature values and append it to the feature vector            
        for feature in feature_table.itertuples():                   
            tokenizer = feature[3]                                              
            sim_fn = feature[4]          
            time_elapsed = 0                                       
            if tokenizer is None:                                               
                start_time = time.time()
                score = sim_fn(l_string, r_string)
                time_elapsed = time.time() - start_time                         
            else:                                           
                start_time = time.time()                    
                score = sim_fn(tokenizer.tokenize(l_string),                  
                               tokenizer.tokenize(r_string))                 
                time_elapsed = time.time() - start_time
            feature_costs[feature[0]] += time_elapsed
                                                                
        if show_progress:                                                       
            prog_bar.update()                                  

    max_cost = max(feature_costs.values())
    for feat_name in feature_costs.keys():
        feature_costs[feat_name] /= max_cost

    feature_table['cost'] = pd.Series(feature_costs.values(), 
                                      feature_costs.keys()) 
    return True 
