
from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP                   
import py_stringsimjoin as ssj     


class Predicate:                                                                
    def __init__(self, feat_name, sim_measure_type, tokenizer_type, sim_function,          
                 tokenizer, comp_op, threshold, cost):                                
        self.feat_name = feat_name
        self.sim_measure_type = sim_measure_type                                
        self.tokenizer_type = tokenizer_type                                    
        self.sim_function = sim_function                                        
        self.tokenizer = tokenizer                                              
        self.comp_op = comp_op                                                  
        self.threshold = threshold                                              
        self.comp_fn = COMP_OP_MAP[self.comp_op]
        self.cost = cost                               

    def set_name(self, name):                                                   
        self.name = name         

    def set_cost(self, cost):
        self.cost = cost

    def set_coverage(self, coverage):
        self.coverage = coverage 
        return True
                                                          
    def is_valid_join_predicate(self):                                                 
        if self.sim_measure_type in ['JACCARD', 'COSINE', 'DICE', 'OVERLAP',   
                                     'OVERLAP_COEFFICIENT']:                   
            return self.comp_op in ['>', '>=', '=']                            
        elif self.sim_measure_type == 'EDIT_DISTANCE':                         
            return self.comp_op in ['<', '<=', '=']                            
        return False    
 
    def apply_pair(self, string1, string2):                                     
        val1 = string1                                                          
        val2 = string2                                                          
        if self.tokenizer_type is not None:                                     
            val1 = self.tokenizer.tokenize(val1)                                
            val2 = self.tokenizer.tokenize(val2)                                
        return self.comp_fn(self.sim_function(val1, val2), self.threshold)      
                                                                                
    def apply_tables(self, ltable, rtable, l_key_attr, r_key_attr,              
                     l_match_attr, r_match_attr, n_jobs=1):                     
        if self.sim_measure_type == 'JACCARD':                                  
            return ssj.jaccard_join(ltable, rtable, l_key_attr, r_key_attr,     
                                    l_match_attr, r_match_attr, self.tokenizer, 
                                    self.threshold, comp_op=self.comp_op,       
                                    n_jobs=n_jobs)                              
        elif self.sim_measure_type == 'COSINE':                                 
            return ssj.cosine_join(ltable, rtable, l_key_attr, r_key_attr,      
                                   l_match_attr, r_match_attr, self.tokenizer,  
                                   self.threshold, comp_op=self.comp_op,        
                                   n_jobs=n_jobs)                               
        elif self.sim_measure_type == 'DICE':                                   
            return ssj.dice_join(ltable, rtable, l_key_attr, r_key_attr,        
                                 l_match_attr, r_match_attr, self.tokenizer,    
                                 self.threshold, comp_op=self.comp_op,          
                                 n_jobs=n_jobs)                                 
        elif self.sim_measure_type == 'EDIT_DISTANCE':                          
            return ssj.edit_distance_join(ltable, rtable,                       
                                          l_key_attr, r_key_attr,               
                                          l_match_attr, r_match_attr,           
                                          self.threshold, comp_op=self.comp_op, 
                                          n_jobs=n_jobs)                        
        elif self.sim_measure_type == 'OVERLAP':                                
            return ssj.overlap_join(ltable, rtable, l_key_attr, r_key_attr,     
                                    l_match_attr, r_match_attr, self.tokenizer, 
                                    self.threshold, comp_op=self.comp_op,       
                                    n_jobs=n_jobs)                              
        elif self.sim_measure_type == 'OVERLAP_COEFFICIENT':                    
            return ssj.overlap_coefficient_join(ltable, rtable,                 
                                    l_key_attr, r_key_attr,                     
                                    l_match_attr, r_match_attr, self.tokenizer, 
                                    self.threshold, comp_op=self.comp_op,       
                                    n_jobs=n_jobs)  
