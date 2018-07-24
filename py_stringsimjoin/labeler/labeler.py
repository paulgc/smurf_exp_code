
def label_table_using_gold(candset, candset_l_key_attr, candset_r_key_attr, 
                           gold_file, label_col='label'):
    gold_pairs = _load_gold_pairs(gold_file)
    labels = (candset[candset_l_key_attr].astype(str) + ',' +           
              candset[candset_r_key_attr].astype(str)).apply(                 
                                      lambda val: gold_pairs.get(val, 0))
    candset[label_col] = labels
    return candset    

def _load_gold_pairs(gold_file):                                                 
    gold_pairs = {}                                                             
    file_handle = open(gold_file, 'r')                                          
    for line in file_handle:                                                    
        gold_pairs[line.strip()] = 1                                            
    file_handle.close()                                                         
    return gold_pairs   
