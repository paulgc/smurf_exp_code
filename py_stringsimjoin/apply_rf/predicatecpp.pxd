
from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "predicatecpp.h" nogil:                                        
    cdef cppclass Predicatecpp nogil:                                           
        Predicatecpp()                                                          
        Predicatecpp(string&, string&, string&, string&, string&, double&)      
        void set_cost(double&)                                                  
        bool is_join_predicate()                                                
        string pred_name, feat_name, sim_measure_type, tokenizer_type, comp_op  
        double threshold, cost     
        bool is_tok_sim_measure
