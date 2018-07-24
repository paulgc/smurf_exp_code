
from py_stringsimjoin.apply_rf.predicate import Predicate
from py_stringsimjoin.apply_rf.rule import Rule
from py_stringsimjoin.apply_rf.rule_set import RuleSet
from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP

 
def extract_pos_rules_from_tree(tree, feature_table, start_rule_id, start_predicate_id):                                         
    feature_names = list(feature_table.index)
    # Get the left, right trees and the threshold from the tree                 
    left = tree.tree_.children_left                                             
    right = tree.tree_.children_right                                           
    threshold = tree.tree_.threshold                                            
                                                                                
    # Get the features from the tree                                            
    features = [feature_names[i] for i in tree.tree_.feature]                   
    value = tree.tree_.value                                                    
                                                    
    rule_set = RuleSet()
#    curr_rule_id = start_rule_id
    #curr_predicate_id = start_predicate_id
                    
    def traverse(node, left, right, features, threshold, depth, cache, start_rule_id, curr_predicate_id):         
        if node == -1:                                                          
            return                                                              
        if threshold[node] != -2:                                               
            # node is not a leaf node
            feat_row = feature_table.ix[features[node]]
            p = Predicate(features[node],
                          feat_row['sim_measure_type'], 
                          feat_row['tokenizer_type'],
                          feat_row['sim_function'], 
                          feat_row['tokenizer'], '<=', threshold[node], feat_row['cost'])                                           
            p.set_name(features[node]+' <= '+str(threshold[node]))                              
            curr_predicate_id += 1 
            cache.insert(depth, p)   
            traverse(left[node], left, right, features, threshold, depth+1, cache, start_rule_id, curr_predicate_id)
            prev_pred = cache.pop(depth)
            feat_row = feature_table.ix[features[node]]                         
            p = Predicate(features[node],
                          feat_row['sim_measure_type'],                         
                          feat_row['tokenizer_type'],                           
                          feat_row['sim_function'],                             
                          feat_row['tokenizer'], '>', threshold[node], feat_row['cost'])                                         
            p.set_name(features[node]+' > '+str(threshold[node]))
            curr_predicate_id += 1
            cache.insert(depth, p)    
            traverse(right[node], left, right, features, threshold, depth+1, cache, start_rule_id, curr_predicate_id)
            prev_pred = cache.pop(depth)                                        
        else:                                                                   
            # node is a leaf node                                               
            if value[node][0][0] <= value[node][0][1]:
                r = Rule(cache[0:depth])
                r.set_name('r'+str(start_rule_id + len(rule_set.rules)+1))
                rule_set.add_rule(r)                                                                        
                print 'pos rule: ', cache[0:depth]                              
                                                                                
    traverse(0, left, right, features, threshold, 0, [], start_rule_id, start_predicate_id)
    return rule_set 

def extract_pos_rules_from_rf(rf, feature_table):
    rule_sets = []
    rule_id = 1
    predicate_id = 1
    tree_id = 1                                                              
    for dt in rf.estimators_:                                                   
        rs = extract_pos_rules_from_tree(dt, feature_table, rule_id, predicate_id)
        rs.set_name('t'+str(tree_id))
        tree_id += 1                                                            
        rule_id += len(rs.rules)
        predicate_id += sum(map(lambda r: len(r.predicates), rs.rules))
        rule_sets.append(rs) 
    return rule_sets
