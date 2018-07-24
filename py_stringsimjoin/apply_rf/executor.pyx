
import time
import random
import pandas as pd
from cython.parallel import prange                                              

from libcpp.vector cimport vector
from libcpp.set cimport set as oset
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map as omap
from libcpp cimport bool                                                        
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE, sprintf
from libc.stdlib cimport atoi                                                   

from py_stringsimjoin.apply_rf.predicate import Predicate                       
from py_stringsimjoin.apply_rf.execution_plan import get_predicate_dict
from py_stringsimjoin.apply_rf.tokenizers cimport tokenize, load_tok, tokenize_str
from py_stringsimjoin.apply_rf.set_sim_join cimport set_sim_join, set_sim_join1
from py_stringsimjoin.apply_rf.overlap_coefficient_join cimport ov_coeff_join                
from py_stringsimjoin.apply_rf.edit_distance_join cimport ed_join   
from py_stringsimjoin.apply_rf.sim_functions cimport cosine, dice, jaccard      
from py_stringsimjoin.apply_rf.utils cimport compfnptr, str_simfnptr, \
  token_simfnptr, get_comp_type, get_comparison_function, get_sim_type, \
  get_str_sim_function, get_token_sim_function, simfnptr_str, get_sim_function_str
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp                
from py_stringsimjoin.apply_rf.node cimport Node                                
from py_stringsimjoin.apply_rf.coverage cimport Coverage         
from py_stringsimjoin.apply_rf.rule cimport Rule                                
from py_stringsimjoin.apply_rf.tree cimport Tree  
from py_stringsimjoin.apply_rf.ex_plan cimport get_default_execution_plan, generate_ex_plan_for_stage2, compute_predicate_cost_and_coverage, extract_pos_rules_from_rf, generate_local_optimal_plans, generate_overall_plan  

cdef extern from "string.h" nogil:                                              
    char *strtok (char *inp_str, const char *delimiters)     

cdef void load_strings(data_path, attr, vector[string]& strings):
    df = pd.read_csv(data_path)
    convert_to_vector1(df[attr], strings)

def test_execute_rf(rf, feature_table, l1, l2, path1, attr1, path2, attr2, working_dir, n_jobs):
    start_time = time.time()                                                    
    cdef vector[Tree] trees, trees1, trees2                                     
    trees = extract_pos_rules_from_rf(rf, feature_table)                        
                                                                                
    cdef int i=0, num_total_trees = trees.size()      
                                                                                
    cdef vector[string] lstrings, rstrings                                      
    load_strings(path1, attr1, lstrings)                                        
    load_strings(path2, attr2, rstrings)                                        
                                                                                
    cdef omap[string, Coverage] coverage                                        
    cdef omap[int, Coverage] tree_cov
    cdef vector[string] l, r                                                    
    for s in l1:                                                                
        l.push_back(lstrings[int(s)])                                       
    for s in l2:                                                                
        r.push_back(rstrings[int(s)])                                       
    print 'computing coverage'                                                  
    compute_predicate_cost_and_coverage(l, r, trees, coverage, tree_cov)                 
    cdef Node global_plan, join_node
    global_plan = get_default_execution_plan(trees, coverage, tree_cov, 
                                             l.size(), trees1, trees2) 

    print 'num join nodes : ', global_plan.children.size()
    for join_node in global_plan.children:                                             
         print 'JOIN', join_node.predicates[0].pred_name

    
    print 'tokenizing strings'                                                  
    tokenize_strings(trees, lstrings, rstrings, working_dir)                    
    print 'finished tokenizing. executing plan'                                 

    execute_plan(global_plan, trees1, lstrings, rstrings, working_dir, n_jobs)  
                                                                                
    cdef pair[vector[pair[int, int]], vector[int]] candset_votes                
    candset_votes = merge_candsets(num_total_trees, trees1,        
                                   working_dir)                                 
    ''' 
    cdef int sample_size = 5000                                                 
    print 'generating plan'
    cdef vector[Node] plans                                                     
    plans = generate_ex_plan_for_stage2(candset_votes.first,                    
                                        lstrings, rstrings,   
                                        trees2, sample_size)  
    print 'executing remaining trees'                                           
    cdef int label = 1, num_trees_processed=trees1.size()                                                          
    i = 0                                                                       
    while candset_votes.first.size() > 0 and i < plans.size():                  
        candset_votes = execute_tree_plan(candset_votes, lstrings, rstrings, plans[i],
                                  num_total_trees, num_trees_processed, label,  
                                  n_jobs, working_dir)                          
        num_trees_processed += 1                                                
        label += 1                                                              
    '''
    print 'total time : ', time.time() - start_time   
  

def execute_rf(rf, feature_table, l1, l2, path1, attr1, path2, attr2, working_dir, n_jobs):                                          
    start_time = time.time()
    cdef vector[Tree] trees, trees1, trees2                                                     
    trees = extract_pos_rules_from_rf(rf, feature_table)

    cdef int i=0, num_total_trees = trees.size(), num_trees_processed                
    num_trees_processed = (num_total_trees / 2) + 1 
#    num_trees_processed = 3
    while i < num_trees_processed:
        trees1.push_back(trees[i])
        i += 1
    
    while i < num_total_trees:
        trees2.push_back(trees[i])
        i += 1
    print 'trees1 : ', trees1.size()
    print 'trees2 : ', trees2.size()
    print 'num trees : ', trees.size()                                          
    num_rules = 0                                                               
    num_preds = 0                                                               
    cdef Tree tree                                                              
    cdef Rule rule                                                              
    for tree in trees:                                                          
        num_rules += tree.rules.size()                                          
        for rule in tree.rules:                                                 
            num_preds += rule.predicates.size()                                 
    print 'num rules : ', num_rules                                             
    print 'num preds : ', num_preds  

    cdef vector[string] lstrings, rstrings                                      
    load_strings(path1, attr1, lstrings)                                        
    load_strings(path2, attr2, rstrings)  

    cdef omap[string, Coverage] coverage
    cdef vector[string] l, r                                                    
    for s in l1:                                                                
        l.push_back(lstrings[int(s) - 1])                                                          
    for s in l2:                                                                
        r.push_back(rstrings[int(s) - 1])
    print 'computing coverage'                                           
#    compute_predicate_cost_and_coverage(l, r, trees1, coverage)    

    cdef vector[Node] plans                                                     
#    generate_local_optimal_plans(trees1, coverage, l.size(), plans)              
#    print 'num pl : ', plans.size()     

    cdef Node global_plan
    global_plan = generate_overall_plan(plans)                                                

    print 'tokenizing strings'
    tokenize_strings(trees, lstrings, rstrings, working_dir)                    
    print 'finished tokenizing. executing plan'
    execute_plan(global_plan, trees1, lstrings, rstrings, working_dir, n_jobs)           

    cdef pair[vector[pair[int, int]], vector[int]] candset_votes                
    candset_votes = merge_candsets(num_total_trees, trees1,        
                                   working_dir)                                 
    cdef int sample_size = 5000
    print 'generating plan'
    plans = generate_ex_plan_for_stage2(candset_votes.first,              
                                                          lstrings, rstrings,   
                                                          trees2, sample_size)  
    print 'executing remaining trees'                                                                            
    cdef int label = 1
    i = 0                                                              
    while candset_votes.first.size() > 0 and i < plans.size():                  
        candset_votes = execute_tree_plan(candset_votes, lstrings, rstrings, plans[i], 
                                  num_total_trees, num_trees_processed, label,
                                  n_jobs, working_dir)  
        num_trees_processed += 1
        label += 1
    print 'total time : ', time.time() - start_time

cdef void execute_plan(Node& root, vector[Tree]& trees, vector[string]& lstrings, 
        vector[string]& rstrings, const string& working_dir, int n_jobs):
#    tokenize_strings(trees, lstrings, rstrings, working_dir)

    cdef pair[vector[pair[int, int]], vector[double]] candset
#    cdef Node root
    print root.children.size(), root.children[0].children.size()

    cdef Node join_node, child_node, curr_node
    print root.node_type, root.predicates.size(), root.children.size()
    for join_node in root.children:
         print 'JOIN', join_node.predicates[0].sim_measure_type, join_node.predicates[0].tokenizer_type, join_node.predicates[0].comp_op, join_node.predicates[0].threshold
         candset = execute_join_node(lstrings, rstrings, join_node.predicates[0], 
                                     n_jobs, working_dir)
         print 'join completed. starting subtree execution.'                                                 
         execute_join_subtree(candset.first, candset.second, lstrings, rstrings, join_node, n_jobs, working_dir) 
         print 'join subtree execution completed'


cdef pair[vector[pair[int, int]], vector[int]] execute_join_subtree(               
                    vector[pair[int, int]]& candset,
                    vector[double]& feature_values,   
                    vector[string]& lstrings, vector[string]& rstrings,            
                    Node& join_subtree, int n_jobs, const string& working_dir):          
    cdef Node child_node, grand_child_node, curr_node                           
                                                                                
    cdef vector[pair[Node, int]] queue                                          
    cdef pair[Node, int] curr_entry                                             
    cdef vector[int] pair_ids, curr_pair_ids                   
                                                                                
    for child_node in join_subtree.children:                                            
        queue.push_back(pair[Node, int](child_node, -1))                        
                                                                                
    cdef omap[int, vector[int]] cached_pair_ids                                 
    cdef omap[int , int] cache_usage                                            
    cdef int curr_index = 0                                                     
    cdef bool top_level_node = False                                            
    cdef vector[double] curr_feature_values
                                                                                
    while queue.size() > 0:                                                     
        curr_entry = queue.back()                                               
        queue.pop_back();                                                       
        curr_node = curr_entry.first                                            

        top_level_node = False
                                                            
        if curr_entry.second == -1:                                             
            top_level_node = True                                               
        else:                                                                   
            pair_ids = cached_pair_ids[curr_entry.second]                       
            cache_usage[curr_entry.second] -= 1                                 
                                                                                
            if cache_usage[curr_entry.second]  == 0:                            
                cache_usage.erase(curr_entry.second)                            
                cached_pair_ids.erase(curr_entry.second)                        
       
        if top_level_node and curr_node.node_type.compare("SELECT") == 0:
            print 'SELECT', curr_node.predicates[0].sim_measure_type, curr_node.predicates[0].tokenizer_type, curr_node.predicates[0].comp_op, curr_node.predicates[0].threshold
            curr_pair_ids = execute_select_node_candset(candset.size(), 
                                    feature_values, curr_node.predicates[0])    
                                                                                
            for child_node in curr_node.children:                     
               queue.push_back(pair[Node, int](child_node, curr_index))
                                                                                
            cache_usage[curr_index] = curr_node.children.size()             
            cached_pair_ids[curr_index] = curr_pair_ids                        
            curr_index += 1
            continue                       

        while (curr_node.node_type.compare("OUTPUT") != 0 and                   
               curr_node.node_type.compare("FILTER") == 0 and                   
               curr_node.children.size() < 2):
            print 'FILTER', curr_node.predicates[0].sim_measure_type, curr_node.predicates[0].tokenizer_type, curr_node.predicates[0].comp_op, curr_node.predicates[0].threshold
            pair_ids = execute_filter_node1(candset, pair_ids, top_level_node,
                                            lstrings, rstrings,                 
                                            curr_node.predicates[0], n_jobs, working_dir)
            curr_node = curr_node.children[0]                                   
            top_level_node = False                                              
                                                                                
        if curr_node.node_type.compare("OUTPUT") == 0:
            if top_level_node:
                write_candset(candset, curr_node.tree_id, curr_node.rule_id,
                              working_dir)    
            else:
                write_candset_using_pair_ids(candset, pair_ids, 
                                             curr_node.tree_id, 
                                             curr_node.rule_id, working_dir)                                   
            continue                                                            
                                                                                
        if curr_node.node_type.compare("FEATURE") == 0:                         
           print 'FEATURE', curr_node.predicates[0].sim_measure_type            
           curr_feature_values = execute_feature_node(candset, pair_ids, top_level_node,
                                                 lstrings, rstrings,            
                                                curr_node.predicates[0], n_jobs,
                                                working_dir)                    
           for child_node in curr_node.children:                                
               print 'SELECT', child_node.predicates[0].sim_measure_type, child_node.predicates[0].tokenizer_type, child_node.predicates[0].comp_op, child_node.predicates[0].threshold
               if top_level_node:
                   curr_pair_ids = execute_select_node_candset(candset.size(), curr_feature_values,
                                                   child_node.predicates[0])   
               else:
                   curr_pair_ids = execute_select_node(pair_ids, curr_feature_values,    
                                                   child_node.predicates[0])    

               for grand_child_node in child_node.children:                     
                   queue.push_back(pair[Node, int](grand_child_node, curr_index))
                                                                                
               cache_usage[curr_index] = child_node.children.size()             
               cached_pair_ids[curr_index] = curr_pair_ids                      
               curr_index += 1                                                  
        elif curr_node.node_type.compare("FILTER") == 0:                        
            print 'FILTER', curr_node.predicates[0].sim_measure_type, curr_node.predicates[0].tokenizer_type, curr_node.predicates[0].comp_op, curr_node.predicates[0].threshold
            pair_ids = execute_filter_node1(candset, pair_ids, top_level_node,
                                           lstrings, rstrings,                  
                                           curr_node.predicates[0], n_jobs, working_dir)
            for child_node in curr_node.children:                               
                queue.push_back(pair[Node, int](child_node, curr_index))        
                                                                                
            cache_usage[curr_index] = curr_node.children.size()                 
            cached_pair_ids[curr_index] = pair_ids                      
            curr_index += 1                                                     


cdef pair[vector[pair[int, int]], vector[int]] merge_candsets(
                                           int num_total_trees, 
                                           vector[Tree]& processed_trees,
                                           const string& working_dir):

    cdef int i=0
    cdef string string_pair
    cdef oset[string] curr_pairs
    cdef omap[string, int] merged_candset
    cdef pair[string, int] entry
    cdef Tree tree
    for tree in processed_trees:
        file_name = working_dir + "/tree_" + str(tree.tree_id)
        print file_name
        f = open(file_name, 'r')
        for line in f:
            curr_pairs.insert(line)
        f.close()
        for string_pair in curr_pairs:
            merged_candset[string_pair] += 1
        curr_pairs.clear()
    cnt = 0
    cdef vector[pair[int, int]] candset_to_be_processed, output_pairs
    cdef vector[int] votes, pair_id

    for entry in merged_candset:
        pair_id = split(entry.first)
        if <double>entry.second >= (<double>num_total_trees/2.0):
            output_pairs.push_back(pair[int, int](pair_id[0], pair_id[1]))
        else:
            candset_to_be_processed.push_back(pair[int, int](pair_id[0], 
                                                             pair_id[1]))
            votes.push_back(entry.second)      
        
    write_output_pairs(output_pairs, working_dir, 0)    

    return pair[vector[pair[int, int]], vector[int]](candset_to_be_processed, 
                                                     votes)


cdef void write_candset(vector[pair[int,int]]& candset, int tree_id, int rule_id, const string& working_dir):
    file_path = working_dir + "/tree_" + str(tree_id)
    f = open(file_path, 'a+')
    cdef pair[int, int] tuple_pair
    for tuple_pair in candset:
        s = str(tuple_pair.first) + ',' + str(tuple_pair.second)
        f.write(s + '\n') 
    f.close()

cdef void write_candset_using_pair_ids(vector[pair[int,int]]& candset, vector[int]& pair_ids, 
                                       int tree_id, int rule_id, const string& working_dir):
    file_path = working_dir + "/tree_" + str(tree_id) 
    f = open(file_path, 'a+')                                                   
    cdef pair[int, int] tuple_pair                                              
    cdef int pair_id
    for pair_id in pair_ids:
        tuple_pair = candset[pair_id]                                                  
        s = str(tuple_pair.first) + ',' + str(tuple_pair.second)                
        f.write(s + '\n')                                                       
    f.close()   

cdef void write_output_pairs(vector[pair[int,int]]& output_pairs, const string& working_dir, int label):
    file_path = working_dir + "/output_" + str(label)
    f = open(file_path, 'w')                                                    
    cdef pair[int, int] tuple_pair                                              
    for tuple_pair in output_pairs:                                                  
        s = str(tuple_pair.first) + ',' + str(tuple_pair.second)                
        f.write(s + '\n')                                                       
    f.close()       

cdef pair[vector[pair[int, int]], vector[double]] execute_join_node(vector[string]& lstrings, vector[string]& rstrings,
                            Predicatecpp predicate, int n_jobs, const string& working_dir):
    cdef vector[vector[int]] ltokens, rtokens

    cdef pair[vector[pair[int, int]], vector[double]] output

    if predicate.sim_measure_type.compare('COSINE') == 0:
        load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)           
        output = set_sim_join(ltokens, rtokens, 0, predicate.threshold, n_jobs)
    elif predicate.sim_measure_type.compare('DICE') == 0:
        load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)       
        output = set_sim_join(ltokens, rtokens, 1, predicate.threshold, n_jobs)                   
    elif predicate.sim_measure_type.compare('JACCARD') == 0:
        load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)       
        output = set_sim_join(ltokens, rtokens, 2, predicate.threshold, n_jobs)                   
    elif predicate.sim_measure_type.compare('OVERLAP_COEFFICIENT') == 0:
        load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)       
        output = ov_coeff_join(ltokens, rtokens, predicate.threshold, n_jobs)
    elif predicate.sim_measure_type.compare('EDIT_DISTANCE') == 0:
        load_tok('qg2_bag', working_dir, ltokens, rtokens)       
        output = ed_join(ltokens, rtokens, 2, predicate.threshold, 
                         lstrings, rstrings, n_jobs)
    return output

cdef vector[pair[int, int]] execute_filter_node(vector[pair[int, int]]& candset, 
                            vector[string]& lstrings, vector[string]& rstrings,
                            Predicatecpp predicate, int n_jobs, const string& working_dir):
    cdef vector[vector[int]] ltokens, rtokens
    if predicate.is_tok_sim_measure:
        print 'before tok'                                   
        load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)           
        print 'loaded tok'                                                                            
    cdef vector[pair[int, int]] partitions, final_output_pairs, part_pairs                    
    cdef vector[vector[pair[int, int]]] output_pairs                            
    cdef int n = candset.size(), start=0, end, i

    partition_size = <int>(<float> n / <float> n_jobs)                           

    for i in range(n_jobs):                                                      
        end = start + partition_size                                            
        if end > n or i == n_jobs - 1:                                           
            end = n                                                             
        partitions.push_back(pair[int, int](start, end))                        
                                                   
        start = end                                                             
        output_pairs.push_back(vector[pair[int, int]]())    

    cdef int sim_type, comp_type                                                     
                                                                                
    sim_type = get_sim_type(predicate.sim_measure_type)                       
    comp_type = get_comp_type(predicate.comp_op)     
    print 'parallen begin'
    for i in prange(n_jobs, nogil=True):                                         
        execute_filter_node_part(partitions[i], candset, ltokens, rtokens,
                                 lstrings, rstrings, 
                                 predicate, sim_type, comp_type, output_pairs[i])
    print 'parallen end'
    for part_pairs in output_pairs:                                             
        final_output_pairs.insert(final_output_pairs.end(), part_pairs.begin(), part_pairs.end())
                                                                             
    return final_output_pairs 

cdef void execute_filter_node_part(pair[int, int] partition,
                                   vector[pair[int, int]]& candset,                           
                                   vector[vector[int]]& ltokens, 
                                   vector[vector[int]]& rtokens,          
                                   vector[string]& lstrings,
                                   vector[string]& rstrings,             
                                   Predicatecpp& predicate,
                                   int sim_type, int comp_type, 
                                   vector[pair[int, int]]& output_pairs) nogil:

    cdef pair[int, int] cand                         
    cdef int i                           
    
    cdef token_simfnptr token_sim_fn
    cdef str_simfnptr str_sim_fn
    cdef compfnptr comp_fn = get_comparison_function(comp_type)

    if predicate.is_tok_sim_measure: 
        token_sim_fn = get_token_sim_function(sim_type)                                                     
        for i in range(partition.first, partition.second):                                                        
            cand  = candset[i]
            if comp_fn(token_sim_fn(ltokens[cand.first], rtokens[cand.second]), 
                       predicate.threshold):
                output_pairs.push_back(cand)         
    else:
        str_sim_fn = get_str_sim_function(sim_type)
        for i in range(partition.first, partition.second):                      
            cand  = candset[i]                                                  
            if comp_fn(str_sim_fn(lstrings[cand.first], rstrings[cand.second]), 
                       predicate.threshold):                                    
                output_pairs.push_back(cand)                 

         
cdef pair[vector[pair[int, int]], vector[int]] execute_tree_plan(
                    pair[vector[pair[int, int]], vector[int]]& candset_votes, 
                    vector[string]& lstrings, vector[string]& rstrings,
                    Node& plan, int num_total_trees, int num_trees_processed,
                    int label, int n_jobs, const string& working_dir):
    cdef Node child_node, grand_child_node, curr_node

    cdef vector[pair[Node, int]] queue
    cdef pair[Node, int] curr_entry
    cdef vector[int] pair_ids, curr_pair_ids, output_pair_ids

    for child_node in plan.children:
        queue.push_back(pair[Node, int](child_node, -1))   
   
    cdef omap[int, vector[int]] cached_pair_ids
    cdef omap[int , int] cache_usage
    cdef int curr_index = 0
    cdef bool top_level_node = False
    cdef vector[double] feature_values

    while queue.size() > 0:
        curr_entry = queue.back()
        queue.pop_back();
        curr_node = curr_entry.first                                        
        
        top_level_node = False

        if curr_entry.second == -1:
            top_level_node = True
        else:
            pair_ids = cached_pair_ids[curr_entry.second]
            cache_usage[curr_entry.second] -= 1
                
            if cache_usage[curr_entry.second]  == 0:
                cache_usage.erase(curr_entry.second)
                cached_pair_ids.erase(curr_entry.second)

        while (curr_node.node_type.compare("OUTPUT") != 0 and
               curr_node.node_type.compare("FILTER") == 0 and 
               curr_node.children.size() < 2):
            print 'FILTER', curr_node.predicates[0].sim_measure_type, curr_node.predicates[0].tokenizer_type, curr_node.predicates[0].comp_op, curr_node.predicates[0].threshold
            pair_ids = execute_filter_node1(candset_votes.first, pair_ids, top_level_node,
                                            lstrings, rstrings,
                                            curr_node.predicates[0], n_jobs, working_dir)
            curr_node = curr_node.children[0]
            top_level_node = False
        
        if curr_node.node_type.compare("OUTPUT") == 0:
            output_pair_ids.insert(output_pair_ids.end(), pair_ids.begin(), 
                                                          pair_ids.end())
            continue

        if curr_node.node_type.compare("FEATURE") == 0:
           print 'FEATURE', curr_node.predicates[0].sim_measure_type
           feature_values = execute_feature_node(candset_votes.first, pair_ids, top_level_node,
                                                 lstrings, rstrings,
                                                curr_node.predicates[0], n_jobs,
                                                working_dir)
           for child_node in curr_node.children:
               print 'SELECT', child_node.predicates[0].sim_measure_type, child_node.predicates[0].tokenizer_type, child_node.predicates[0].comp_op, child_node.predicates[0].threshold

               if top_level_node:
                   curr_pair_ids = execute_select_node_candset(candset_votes.first.size(), feature_values,    
                                                       child_node.predicates[0])  
               else:
                   curr_pair_ids = execute_select_node(pair_ids, feature_values, 
                                                   child_node.predicates[0])

               for grand_child_node in child_node.children:
                   queue.push_back(pair[Node, int](grand_child_node, curr_index))
              
               cache_usage[curr_index] = child_node.children.size()
               cached_pair_ids[curr_index] = curr_pair_ids                                 
               curr_index += 1         
        elif curr_node.node_type.compare("FILTER") == 0:
            print 'FILTER', curr_node.predicates[0].sim_measure_type, curr_node.predicates[0].tokenizer_type, curr_node.predicates[0].comp_op, curr_node.predicates[0].threshold
            pair_ids = execute_filter_node1(candset_votes.first, pair_ids, top_level_node,
                                           lstrings, rstrings,      
                                           curr_node.predicates[0], n_jobs, working_dir)           
            for child_node in curr_node.children:
                queue.push_back(pair[Node, int](child_node, curr_index))

            cache_usage[curr_index] = curr_node.children.size()
            cached_pair_ids[curr_index] = pair_ids
            curr_index += 1

    cdef int pair_id
    for pair_id in output_pair_ids:
        candset_votes.second[pair_id] += 1
    
    cdef vector[pair[int, int]] next_candset, output_pairs
    cdef vector[int] next_votes
    cdef int curr_votes
    cdef double reqd_votes = (<double>num_total_trees)/2.0
    for i in xrange(candset_votes.second.size()):
        curr_votes = candset_votes.second[i]
        if curr_votes + num_total_trees - num_trees_processed - 1 < reqd_votes:
            continue
        if curr_votes >= reqd_votes:
            output_pairs.push_back(candset_votes.first[i])
        else:
            next_candset.push_back(candset_votes.first[i])
            next_votes.push_back(curr_votes)
 
    write_output_pairs(output_pairs, working_dir, label)                            
                                                                                
    return pair[vector[pair[int, int]], vector[int]](next_candset,   
                                                     next_votes)           
            

cdef vector[double] execute_feature_node(vector[pair[int, int]]& candset, 
                                         vector[int]& pair_ids, 
                                         bool top_level_node,
                                         vector[string]& lstrings, 
                                         vector[string]& rstrings,
                                         Predicatecpp predicate, 
                                         int n_jobs, const string& working_dir):
    cdef vector[vector[int]] ltokens, rtokens                                   

    if predicate.is_tok_sim_measure:
        print 'before tok'                                                          
        load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)           
        print 'loaded tok'                                                          

    cdef int n, sim_type, i        
    
    if top_level_node:
        n = candset.size()
    else:
        n = pair_ids.size()

    cdef vector[double] feature_values = xrange(0, n)                                                                                    
                                                                                
    sim_type = get_sim_type(predicate.sim_measure_type)                         
    cdef token_simfnptr token_sim_fn
    cdef str_simfnptr str_sim_fn                           
    cdef pair[int, int] cand

    if predicate.is_tok_sim_measure:
        token_sim_fn = get_token_sim_function(sim_type)                                 
        if top_level_node:                                                
            for i in prange(n, nogil=True, num_threads=n_jobs):
                cand = candset[i]
                feature_values[i] = token_sim_fn(ltokens[cand.first], 
                                                 rtokens[cand.second]) 
        else:
            for i in prange(n, nogil=True, num_threads=n_jobs):                     
                cand = candset[pair_ids[i]]                                                   
                feature_values[i] = token_sim_fn(ltokens[cand.first], 
                                                 rtokens[cand.second])
    else:
        str_sim_fn = get_str_sim_function(sim_type)                         
        if top_level_node:                                                      
            for i in prange(n, nogil=True, num_threads=n_jobs):                 
                cand = candset[i]                                               
                feature_values[i] = str_sim_fn(lstrings[cand.first],           
                                               rstrings[cand.second])          
        else:                                                                   
            for i in prange(n, nogil=True, num_threads=n_jobs):                 
                cand = candset[pair_ids[i]]                                     
                feature_values[i] = str_sim_fn(lstrings[cand.first],           
                                               rstrings[cand.second])                                                                        
    return feature_values                                            


cdef vector[int] execute_select_node(vector[int]& pair_ids,                 
                                     vector[double]& feature_values,              
                                     Predicatecpp& predicate):            
    cdef vector[int] output_pair_ids                              
    cdef int n = pair_ids.size(), pair_id, comp_type                                   
                                                                                
    comp_type = get_comp_type(predicate.comp_op)                         
    cdef compfnptr comp_fn = get_comparison_function(comp_type)                           
                                                                                
    for i in xrange(n):                         
        if comp_fn(feature_values[i], predicate.threshold):
            output_pair_ids.push_back(pair_ids[i])
                                                                                
    return output_pair_ids

cdef vector[int] execute_select_node_candset(int n,                     
                                             vector[double]& feature_values,            
                                             Predicatecpp& predicate):                  
    cdef vector[int] output_pair_ids                                            
    cdef pair_id, comp_type                            
                                                                                
    comp_type = get_comp_type(predicate.comp_op)                                
    cdef compfnptr comp_fn = get_comparison_function(comp_type)                 
                                                                                
    for pair_id in xrange(n):                                                         
        if comp_fn(feature_values[pair_id], predicate.threshold):                     
            output_pair_ids.push_back(pair_id)                              
                                                                                
    return output_pair_ids         

cdef vector[int] execute_filter_node1(vector[pair[int, int]]& candset, 
                                     vector[int]& pair_ids,
                                     bool top_level_node,
                                     vector[string]& lstrings, 
                                     vector[string]& rstrings,
                                     Predicatecpp predicate, 
                                     int n_jobs, const string& working_dir):
    cdef vector[vector[int]] ltokens, rtokens                                   
    if predicate.is_tok_sim_measure:
        print 'before tok'                                                          
        load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)           
        print 'loaded tok'                                                          
    cdef vector[pair[int, int]] partitions
    cdef vector[int] final_output_pairs, part_pairs      
    cdef vector[vector[int]] output_pairs                            
    cdef int n, start=0, end, i                                

    if top_level_node:
        n = candset.size()
    else:
        n = pair_ids.size()
                                                                                
    partition_size = <int>(<float> n / <float> n_jobs)                          
                                                                                
    for i in range(n_jobs):                                                     
        end = start + partition_size                                            
        if end > n or i == n_jobs - 1:                                          
            end = n                                                             
        partitions.push_back(pair[int, int](start, end))                        
                                                                                
        start = end                                                             
        output_pairs.push_back(vector[int]())                        
                                                                                
    cdef int sim_type, comp_type                                                
                                                                                
    sim_type = get_sim_type(predicate.sim_measure_type)                         
    comp_type = get_comp_type(predicate.comp_op)                                
    print 'parallen begin'                                                      
    for i in prange(n_jobs, nogil=True):                                        
        execute_filter_node_part1(partitions[i], candset, pair_ids, top_level_node, 
                                  ltokens, rtokens, lstrings, rstrings,      
                                 predicate, sim_type, comp_type, output_pairs[i])
    print 'parallen end'                                                        
    for part_pairs in output_pairs:                                             
        final_output_pairs.insert(final_output_pairs.end(), part_pairs.begin(), part_pairs.end())
                                                                                
    return final_output_pairs                                                   
                                                                                
cdef void execute_filter_node_part1(pair[int, int] partition,                    
                                   vector[pair[int, int]]& candset,
                                   vector[int]& pair_ids,   
                                   bool top_level_node,          
                                   vector[vector[int]]& ltokens,                
                                   vector[vector[int]]& rtokens,
                                   vector[string]& lstrings,
                                   vector[string]& rstrings,                
                                   Predicatecpp& predicate,                     
                                   int sim_type, int comp_type,                 
                                   vector[int]& output_pairs) nogil: 
                                                                                
    cdef pair[int, int] cand                                                    
    cdef int i                                                                  
                                                                                
    cdef str_simfnptr str_sim_fn
    cdef token_simfnptr token_sim_fn                                                                        
    cdef compfnptr comp_fn = get_comparison_function(comp_type)                 

    if predicate.is_tok_sim_measure:
        token_sim_fn = get_token_sim_function(sim_type)                         
        if top_level_node:                 
            for i in range(partition.first, partition.second):                      
                cand  = candset[i]                                                  
                if comp_fn(token_sim_fn(ltokens[cand.first], rtokens[cand.second]), 
                           predicate.threshold):
                    output_pairs.push_back(i)  
        else:                                                               
            for i in range(partition.first, partition.second):
                cand  = candset[pair_ids[i]]                                                      
                if comp_fn(token_sim_fn(ltokens[cand.first], rtokens[cand.second]), 
                           predicate.threshold):
                    output_pairs.push_back(pair_ids[i])
    else:
        str_sim_fn = get_str_sim_function(sim_type)                         
        if top_level_node:                                                      
            for i in range(partition.first, partition.second):                  
                cand  = candset[i]                                              
                if comp_fn(str_sim_fn(lstrings[cand.first], rstrings[cand.second]), 
                           predicate.threshold):                                
                    output_pairs.push_back(i)                                   
        else:                                                                   
            for i in range(partition.first, partition.second):                  
                cand  = candset[pair_ids[i]]                                    
                if comp_fn(str_sim_fn(lstrings[cand.first], rstrings[cand.second]), 
                           predicate.threshold):                                
                    output_pairs.push_back(pair_ids[i])   
           

cdef vector[int] split(string inp_string) nogil:                                      
    cdef char* pch                                                              
    pch = strtok (<char*> inp_string.c_str(), ",")                              
    cdef vector[int] out_tokens                                                 
    while pch != NULL:                                                          
        out_tokens.push_back(atoi(pch))                                         
        pch = strtok (NULL, ",")                                                
    return out_tokens    

cdef void tokenize_strings(vector[Tree]& trees, vector[string]& lstrings, 
                      vector[string]& rstrings, const string& working_dir):
    cdef oset[string] tokenizers
    cdef Tree tree
    cdef Rule rule
    cdef Predicatecpp predicate
    for tree in trees:
        for rule in tree.rules:
            for predicate in rule.predicates:
                if predicate.sim_measure_type.compare('EDIT_DISTANCE') == 0:
                    tokenizers.insert('qg2_bag')
                    continue 
                tokenizers.insert(predicate.tokenizer_type)
 
    cdef string tok_type
    for tok_type in tokenizers:
        tokenize(lstrings, rstrings, tok_type, working_dir)

def test_tok1(df1, attr1, df2, attr2):                                                         
    cdef vector[string] lstrings, rstrings                                                 
    convert_to_vector1(df1[attr1], lstrings)
    convert_to_vector1(df2[attr2], rstrings)
    tokenize(lstrings, rstrings, 'ws', 'gh')                       
  
cdef void convert_to_vector1(string_col, vector[string]& string_vector):         
    for val in string_col:                                                      
        string_vector.push_back(str(val))   

cdef vector[string] infer_tokenizers(plan, rule_sets):
    cdef vector[string] tokenizers
    predicate_dict = get_predicate_dict(rule_sets)

    queue = []
    queue.extend(plan.root.children)
    cdef string s
    while len(queue) > 0:
        curr_node = queue.pop(0)

        if curr_node.node_type in ['JOIN', 'FEATURE', 'FILTER']:
            pred = predicate_dict.get(curr_node.predicate)
            s = pred.tokenizer_type
            tokenizers.push_back(s)
    
        if curr_node.node_type == 'OUTPUT':
            continue
        queue.extend(curr_node.children)
    return tokenizers

def generate_tokens(ft, path1, attr1, path2, attr2, const string& working_dir):
    cdef oset[string] tokenizers                                                
    for idx, row in ft.iterrows():
        if row['sim_measure_type'] == 'EDIT_DISTANCE':
            tokenizers.insert('qg2_bag')
            continue
        tokenizers.insert(str(row['tokenizer_type']))

    cdef vector[string] lstrings, rstrings                                      
    load_strings(path1, attr1, lstrings)                                        
    load_strings(path2, attr2, rstrings)   

    cdef string tok_type                                                        
    for tok_type in tokenizers:                                                 
        tokenize(lstrings, rstrings, tok_type, working_dir)    

def perform_join(path1, attr1, path2, attr2, tok_type, sim_type, threshold, const string& working_dir):
    cdef vector[vector[int]] ltokens, rtokens                                   
    cdef pair[vector[pair[int, int]], vector[double]] output   
    cdef pair[int, int] entry
    cdef vector[string] lstrings, rstrings                                      

    if sim_type == 'COSINE':
        load_tok(tok_type, working_dir, ltokens, rtokens)                           
        threshold = threshold - 0.0001                                              
        output = set_sim_join(ltokens, rtokens, 0, threshold, 4)           
    elif sim_type == 'DICE':
        load_tok(tok_type, working_dir, ltokens, rtokens)                       
        threshold = threshold - 0.0001                                          
        output = set_sim_join(ltokens, rtokens, 1, threshold, 4)   
    elif sim_type == 'JACCARD':
        load_tok(tok_type, working_dir, ltokens, rtokens)                       
        threshold = threshold - 0.0001                                          
        output = set_sim_join(ltokens, rtokens, 2, threshold, 4)           
    elif sim_type == 'OVERLAP_COEFFICIENT':
        load_tok(tok_type, working_dir, ltokens, rtokens)                       
        threshold = threshold - 0.0001
        output = ov_coeff_join(ltokens, rtokens, threshold, 4)
    elif sim_type == 'EDIT_DISTANCE':
        load_tok('qg2_bag', working_dir, ltokens, rtokens)
        load_strings(path1, attr1, lstrings)                                        
        load_strings(path2, attr2, rstrings)                         
        output = ed_join(ltokens, rtokens, 2, threshold, lstrings, rstrings, 4)
   
    output_pairs = []
    for i in xrange(output.first.size()):
        output_pairs.append([str(output.first[i].first) + ',' + str(output.first[i].second), output.second[i]])
    output_df = pd.DataFrame(output_pairs, columns=['pair_id', 'score'])
    return output_df
 
def test_jac(sim_type, threshold):
    st = time.time()
    print 'tokenizing'
    #test_tok1(df1, attr1, df2, attr2)
    print 'tokenizing done.'
    cdef vector[vector[int]] ltokens, rtokens
    cdef vector[pair[int, int]] output
    cdef pair[vector[pair[int, int]], vector[double]] output1
    load_tok('ws', 't5', ltokens, rtokens)
    print 'loaded tok'
    cdef int i
#    for i in xrange(50):
#        print 'i= ', i
#    if sim_type == 3:
    for i in xrange(50):
        output1 = ov_coeff_join(ltokens, rtokens, threshold, 4)             
        print 'output size : ', output.size()                                       
#    else:
#        output1 = set_sim_join(ltokens, rtokens, sim_type, threshold)
#    print 'scores size : ', output1.second.size()

#    cdef pair[int, int] entry
#    for i in xrange(output1.first.size()):
#        if output1.first[i].first == 1954 and output1.first[i].second == 63847:
#            print 'sim score : ', output1.second[i]
    print 'time : ', time.time() - st


def execute_rf_naive(rf, feature_table, ldf, attr1, rdf, attr2):
    cdef vector[string] lstrings, rstrings                                      
    convert_to_vector1(ldf[attr1], lstrings)                                    
    convert_to_vector1(rdf[attr2], rstrings)

    cdef vector[pair[simfnptr_str, string]] feature_info
    cdef oset[string] tokenizers
    for feat_name in feature_table.index:
        feature_info.push_back(pair[simfnptr_str, string](get_sim_function_str(get_sim_type(feature_table.ix[feat_name]['sim_measure_type'])), 
                                                      feature_table.ix[feat_name]['tokenizer_type']))
        tokenizers.insert(feature_table.ix[feat_name]['tokenizer_type'])

    cdef vector[string] tokens1, tokens2
    cdef string tok_type, str1, str2
    cdef int id1=0, id2=0, cnt= 0     
    cdef vector[pair[int, int]] candset
    cdef pair[simfnptr_str, string] entry
    cdef omap[string, vector[string]] ltokens, rtokens
    for str1 in lstrings:
        id2 = 0
        for tok_type in tokenizers:
            ltokens[tok_type] = tokenize_str(str1, tok_type)

        for str2 in rstrings:
            for tok_type in tokenizers:                                              
                rtokens[tok_type] = tokenize_str(str2, tok_type)  
            f = []
            for entry in feature_info:
                f.append(entry.first(ltokens[entry.second], rtokens[entry.second]))
            if rf.predict([f]) == 1:
                 candset.push_back(pair[int, int](id1, id2))
            id2 += 1
            cnt += 1
#            if cnt % 1000000 == 0:
            print cnt
        id1 += 1
                                                            
