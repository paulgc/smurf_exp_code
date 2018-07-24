
from joblib import delayed, Parallel                                            
import pandas as pd                                                             
import pyprind 

from py_stringsimjoin.utils.generic_helper import build_dict_from_table, \
    get_num_processes_to_launch, split_table                                    
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_attr_type, validate_key_attr, validate_input_table


def extract_feature_vecs(candset, candset_l_key_attr, candset_r_key_attr,                       
                         ltable, rtable,                                               
                         l_key_attr, r_key_attr, l_join_attr, r_join_attr,
                         feature_table, n_jobs=1, show_progress=True):
    # check if the input candset is a dataframe                             
    validate_input_table(candset, 'candset')                                
                                                                                
    # check if the candset key attributes exist                             
    validate_attr(candset_l_key_attr, candset.columns,                      
                  'left key attribute', 'candset')                          
    validate_attr(candset_r_key_attr, candset.columns,                      
                  'right key attribute', 'candset')                         
                                                                                
    # check if the input tables are dataframes                              
    validate_input_table(ltable, 'left table')                              
    validate_input_table(rtable, 'right table')                             
                                                                                
    # check if the key attributes and join attributes exist              
    validate_attr(l_key_attr, ltable.columns,                               
                  'key attribute', 'left table')                            
    validate_attr(r_key_attr, rtable.columns,                               
                  'key attribute', 'right table')                           
    validate_attr(l_join_attr, ltable.columns,                            
                  'join attribute', 'left table')                         
    validate_attr(r_join_attr, rtable.columns,                            
                  'join attribute', 'right table')                        
                                                                                
    # check if the join attributes are not of numeric type                      
    validate_attr_type(l_join_attr, ltable[l_join_attr].dtype,          
                       'join attribute', 'left table')                    
    validate_attr_type(r_join_attr, rtable[r_join_attr].dtype,          
                       'join attribute', 'right table')                   
                                                                                
    # check if the key attributes are unique and do not contain missing values                                                        
    validate_key_attr(l_key_attr, ltable, 'left table')                     
    validate_key_attr(r_key_attr, rtable, 'right table')                    
                                                                                
    # Do a projection on the input dataframes to keep only required         
    # attributes. Note that this does not create a copy of the dataframes.  
    # It only creates a view on original dataframes.                        
    ltable_projected = ltable[[l_key_attr, l_join_attr]]                  
    rtable_projected = rtable[[r_key_attr, r_join_attr]]                  
                                                                                
    # computes the actual number of jobs to launch.                         
    n_jobs = min(get_num_processes_to_launch(n_jobs), len(candset))         
                                                                                
    if n_jobs <= 1:                                                         
        # if n_jobs is 1, do not use any parallel code.                     
        output_table =  _extract_feature_vecs_split(candset,                      
                                         candset_l_key_attr, candset_r_key_attr,
                                         ltable_projected, rtable_projected,    
                                         l_key_attr, r_key_attr,                
                                         l_join_attr, r_join_attr,          
                                         feature_table, show_progress)                   
    else:   
        # if n_jobs is above 1, split the candset into n_jobs splits and    
        # filter each candset split in a separate process.                  
        candset_splits = split_table(candset, n_jobs)                       
        results = Parallel(n_jobs=n_jobs)(delayed(_extract_feature_vecs_split)(   
                                      candset_splits[job_index],                
                                      candset_l_key_attr, candset_r_key_attr,   
                                      ltable_projected, rtable_projected,       
                                      l_key_attr, r_key_attr,                   
                                      l_join_attr, r_join_attr,             
                                      feature_table,                                     
                                      (show_progress and (job_index==n_jobs-1)))
                                          for job_index in range(n_jobs))       
        output_table = pd.concat(results)                                   
                                                                                
    return output_table  


def _extract_feature_vecs_split(candset, 
                                candset_l_key_attr, candset_r_key_attr,       
                                ltable, rtable,                                        
                                l_key_attr, r_key_attr, 
                                l_join_attr, r_join_attr,      
                                feature_table, show_progress):
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
    
    feature_vectors = []

    if show_progress:                                                           
        prog_bar = pyprind.ProgBar(len(candset))

    for candset_row in candset.itertuples(index=False):
        l_id = candset_row[candset_l_key_attr_index]                            
        r_id = candset_row[candset_r_key_attr_index]                            
                                                                                
        l_string = str(ltable_dict[l_id][l_join_attr_index])                                              
        r_string = str(rtable_dict[r_id][r_join_attr_index])                                               

        fv = []
        # append key attribute values
        fv.append(candset_row[0])
        fv.append(candset_row[candset_l_key_attr_index])
        fv.append(candset_row[candset_r_key_attr_index]) 

        # compute feature values and append it to the feature vector
        for feature in feature_table.itertuples(index=False):
            tokenizer = feature[2]
            sim_fn = feature[3]  
            if tokenizer is None:
                fv.append(sim_fn(l_string, r_string))
            else:
                fv.append(sim_fn(tokenizer.tokenize(l_string), 
                                 tokenizer.tokenize(r_string)))       

        feature_vectors.append(tuple(fv))                                                                        
                                                                                
        if show_progress:                                                       
            prog_bar.update()

    # obtain the header of the feature vectors table
    feature_vectors_table_header = ['_id', candset_l_key_attr, 
                                    candset_r_key_attr]
    feature_vectors_table_header.extend(list(feature_table.index))

    # create output fetaure vectors table
    feature_vectors_table = pd.DataFrame(feature_vectors,
                                         index=candset.index.values,
                                         columns = feature_vectors_table_header)
    return feature_vectors_table      
