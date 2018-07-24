
import time

from cython.parallel import prange                                              

from libcpp.vector cimport vector
from libcpp.set cimport set as oset
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map as omap
from libcpp cimport bool                                                        
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE, sprintf

from py_stringsimjoin.apply_rf.execution_plan import get_predicate_dict
from py_stringsimjoin.apply_rf.tokenizers cimport tokenize, load_tok
from py_stringsimjoin.apply_rf.set_sim_join cimport set_sim_join
from py_stringsimjoin.apply_rf.overlap_coefficient_join cimport ov_coeff_join                
from py_stringsimjoin.apply_rf.edit_distance_join cimport ed_join   
from py_stringsimjoin.apply_rf.sim_functions cimport cosine, dice, jaccard      
from py_stringsimjoin.apply_rf.utils cimport compfnptr, simfnptr, get_comp_type, get_comparison_function, get_sim_type, get_sim_function

####from py_stringsimjoin.apply_rf import tokenizers

#cdef extern from "<algorithm>" namespace "std":
#    void std_sort "std::sort" [iter](iter first, iter last)
cdef extern from "predicatecpp.h" nogil:                                      
    cdef cppclass Predicatecpp nogil:                                          
        Predicatecpp()                                                         
        Predicatecpp(string&, string&, string&, string&, string&, double&)
        void set_cost(double&)
        string pred_name, feat_name, sim_measure_type, tokenizer_type, comp_op                                                  
        double threshold, cost

cdef extern from "node.h" nogil:                                           
    cdef cppclass Node nogil:                                              
        Node()                                                             
        Node(vector[Predicatecpp]&, string&, vector[Node]&)                           
        vector[Predicatecpp] predicates
        string node_type
        vector[Node] children

cdef Node clone_execution_plan(plan, rule_sets):
    predicate_dict = get_predicate_dict(rule_sets)
   
    cdef Node root = Node(vector[Predicatecpp](), 'ROOT', vector[Node]())
    old_queue = [plan.root]
    cdef vector[Node] new_queue
    new_queue.push_back(root)
    cdef int pos = 0
    cdef Node new_child_node, new_curr_node
    cdef Predicatecpp new_pred
    cdef vector[Predicatecpp] new_preds
    cdef vector[Node] new_child_nodes

    children = {}
 
    while len(old_queue) > 0:
        curr_node = old_queue.pop(0)
        children[pos] = []
        for child_node in curr_node.children:
            new_preds = vector[Predicatecpp]()                              
            if child_node.node_type != 'OUTPUT':
                '''               
                if len(child_node.predicates) > 0:
                    for pred in child_node.predicates:
                        new_pred = Predicatecpp(predicate_dict[pred].sim_measure_type,
                                                predicate_dict[pred].tokenizer_type,
                                                predicate_dict[pred].comp_op,
                                                predicate_dict[pred].threshold)
                        new_preds.push_back(new_pred)
                else:
                '''            
                new_pred = Predicatecpp('','', predicate_dict[child_node.predicate].sim_measure_type,
                                            predicate_dict[child_node.predicate].tokenizer_type,
                                            predicate_dict[child_node.predicate].comp_op,
                                            predicate_dict[child_node.predicate].threshold)
                new_preds.push_back(new_pred)
            new_child_node = Node(new_preds, child_node.node_type, vector[Node]())
            old_queue.append(child_node)
            new_queue.push_back(new_child_node)
            new_queue[pos].children.push_back(new_queue[new_queue.size()-1])
            children[pos].append(new_queue.size()-1)
        pos += 1
    for k in range(pos-1, -1, -1):
        new_queue[k].children = vector[Node]()
        for v in children[k]:
            new_queue[k].children.push_back(new_queue[v])

    print new_queue[0].children.size(), new_queue[1].children.size(), new_queue[0].children[0].children.size()
    return new_queue[0]

def ex_plan(plan, rule_sets, df1, attr1, df2, attr2, working_dir, n_jobs):                                          
    cdef vector[string] lstrings, rstrings                                      
    convert_to_vector1(df1[attr1], lstrings)                                    
    convert_to_vector1(df2[attr2], rstrings)
    execute_plan(plan, rule_sets, lstrings, rstrings, working_dir, n_jobs)           

cdef void execute_plan(plan, rule_sets, vector[string]& lstrings, 
        vector[string]& rstrings, const string& working_dir, int n_jobs):
#    tokenize_strings(plan, rule_sets, lstrings, rstrings, working_dir)

    cdef vector[pair[int, int]] candset, curr_candset
#    cdef Node root
    print 'cloning'
    cdef Node root = clone_execution_plan(plan, rule_sets)
    print 'clone finished'
    print root.children.size(), root.children[0].children.size()
    print len(plan.root.children), len(plan.root.children[0].children)

    cdef Node join_node, child_node, curr_node
    print len(plan.root.children), root.node_type, root.predicates.size(), root.children.size()
    for join_node in root.children:
         print 'JOIN', join_node.predicates[0].sim_measure_type, join_node.predicates[0].tokenizer_type, join_node.predicates[0].comp_op, join_node.predicates[0].threshold
         candset = execute_join_node(lstrings, rstrings, join_node.predicates[0], 
                                     n_jobs, working_dir)
         for child_node in join_node.children:
             curr_node = child_node
             curr_candset = candset

             while curr_node.node_type.compare('OUTPUT') != 0:
                 print 'FILTER', curr_node.predicates[0].sim_measure_type, curr_node.predicates[0].tokenizer_type, curr_node.predicates[0].comp_op, curr_node.predicates[0].threshold
                 curr_candset = execute_filter_node(curr_candset, lstrings, rstrings, 
                                    curr_node.predicates[0], n_jobs, working_dir)
                 print 'filter done'
                 print curr_node.children.size()
                 curr_node = curr_node.children[0]
             
             print 'candset after join : ', candset.size() , " , candset at output : ", curr_candset.size()                        

cdef vector[pair[int, int]] execute_join_node(vector[string]& lstrings, vector[string]& rstrings,
                            Predicatecpp predicate, int n_jobs, const string& working_dir):
    cdef vector[vector[int]] ltokens, rtokens
    load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)               

    cdef vector[pair[int, int]] output

    if predicate.sim_measure_type.compare('COSINE') == 0:
        output = set_sim_join(ltokens, rtokens, 0, predicate.threshold)
    elif predicate.sim_measure_type.compare('DICE') == 0:
        output = set_sim_join(ltokens, rtokens, 1, predicate.threshold)                   
    elif predicate.sim_measure_type.compare('JACCARD') == 0:
        output = set_sim_join(ltokens, rtokens, 2, predicate.threshold)                   
    return output

cdef vector[pair[int, int]] execute_filter_node(vector[pair[int, int]]& candset, vector[string]& lstrings, vector[string]& rstrings,
                            Predicatecpp predicate, int n_jobs, const string& working_dir):
    cdef vector[vector[int]] ltokens, rtokens
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
                                 predicate, sim_type, comp_type, output_pairs[i])
    print 'parallen end'
    for part_pairs in output_pairs:                                             
        final_output_pairs.insert(final_output_pairs.end(), part_pairs.begin(), part_pairs.end())
                                                                             
    return final_output_pairs 

cdef void execute_filter_node_part(pair[int, int] partition,
                                   vector[pair[int, int]]& candset,                           
                                   vector[vector[int]]& ltokens, 
                                   vector[vector[int]]& rtokens,                       
                                   Predicatecpp& predicate,
                                   int sim_type, int comp_type, 
                                   vector[pair[int, int]]& output_pairs) nogil:

    cdef pair[int, int] cand                         
    cdef int i                           
    
    cdef simfnptr sim_fn = get_sim_function(sim_type)
    cdef compfnptr comp_fn = get_comparison_function(comp_type)
                                                      
    for i in range(partition.first, partition.second):                                                        
        cand  = candset[i]
        if comp_fn(sim_fn(ltokens[cand.first], rtokens[cand.second]), predicate.threshold):
            output_pairs.push_back(cand)         
 
cdef void tokenize_strings(plan, rule_sets, vector[string]& lstrings, 
                      vector[string]& rstrings, const string& working_dir):
    tokenizers = infer_tokenizers(plan, rule_sets)
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
        string_vector.push_back(val)   

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

def test_jac(df1, attr1, df2, attr2, sim_type, threshold):
    st = time.time()
    print 'tokenizing'
    #test_tok1(df1, attr1, df2, attr2)
    print 'tokenizing done.'
    cdef vector[vector[int]] ltokens, rtokens
    cdef vector[pair[int, int]] output
    load_tok('ws', 'gh', ltokens, rtokens)
    print 'loaded tok'
    if sim_type == 3:
        output = ov_coeff_join(ltokens, rtokens, threshold)             
    else:
        output = set_sim_join(ltokens, rtokens, sim_type, threshold)
    print 'output size : ', output.size()
    print 'time : ', time.time() - st

def test_ed(df1, attr1, df2, attr2, threshold):                      
    st = time.time()                                                            
    print 'tokenizing'                                                          
    cdef vector[string] lstrings, rstrings                                      
    convert_to_vector1(df1[attr1], lstrings)                                    
    convert_to_vector1(df2[attr2], rstrings)  
    tokenize(lstrings, rstrings, 'qg2', 'gh1')                                    
    print 'tokenizing done.'                                                    
    cdef vector[vector[int]] ltokens, rtokens                                   
    cdef vector[pair[int, int]] output                                          
    load_tok('qg2', 'gh1', ltokens, rtokens)                                      
    print 'loaded tok'                                                          
    output = ed_join(ltokens, rtokens, 2, threshold, lstrings, rstrings)            
    print 'output size : ', output.size()                                       
    print 'time : ', time.time() - st                                           
                                        
