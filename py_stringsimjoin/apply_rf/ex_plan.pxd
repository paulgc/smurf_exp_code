
from libcpp.vector cimport vector             
from libcpp.pair cimport pair                                  
from libcpp.string cimport string
from libcpp.map cimport map as omap                                             

from py_stringsimjoin.apply_rf.tree cimport Tree                                
from py_stringsimjoin.apply_rf.rule cimport Rule
from py_stringsimjoin.apply_rf.node cimport Node                                
from py_stringsimjoin.apply_rf.coverage cimport Coverage    


cdef void compute_predicate_cost_and_coverage(vector[string]& lstrings, 
                                              vector[string]& rstrings, 
                                              vector[Tree]& trees, 
                                              omap[string, Coverage]& coverage,
                                              omap[int, Coverage]& tree_cov)

cdef Node get_default_execution_plan(vector[Tree]& trees, 
                                     omap[string, Coverage]& coverage,
                                     omap[int, Coverage]& tree_cov, const int,
                                     vector[Tree]& sel_trees, vector[Tree]& rem_trees)

cdef vector[Tree] extract_pos_rules_from_rf(rf, feature_table)
cdef void generate_local_optimal_plans(vector[Tree]& trees, omap[string, Coverage]& coverage, int sample_size, vector[Node]& plans, vector[int]& num_join_nodes)
cdef Node generate_overall_plan(vector[Node] plans)

cdef vector[Node] generate_ex_plan_for_stage2(vector[pair[int, int]]&, vector[string]& lstrings,         
                                              vector[string]& rstrings,         
                                              vector[Tree]& trees, int)
