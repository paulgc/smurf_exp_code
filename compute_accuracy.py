import sys

def compute_accuracy(gold_path, output_pairs):
    gold_map = {}                                                                   
    f = open(gold_path, 'r')                                                        
    i = 0                                                                           
    for l in f:                                                                     
        if i == 0:                                                                  
            i = 1                                                                   
            continue                                                                
        gold_map[l.strip()] = True                                                  
    f.close()
    
    tp = 0                                                                          
    total_pairs = 0                                                                 
                                                                                
    for l in output_pairs.keys():                                                   
        total_pairs += 1                                                            
        if gold_map.get(l) is not None:                                             
            tp += 1                                                                 
    print tp, total_pairs                                                                            
    if tp == 0 or total_pairs == 0:
        pr = 1.0
        re = 0
    else:
        pr = float(tp) / float(total_pairs)                                             
        re = float(tp) / float(len(gold_map.keys()))                                    
                                                                                
    f1 = (2.0 * pr * re) / (pr + re)

    return (pr, re, f1)

def compute_accuracy1(gold_path, output_pairs):                                  
    gold_map = {}                                                               
    f = open(gold_path, 'r')                                                    
    i = 0                                                                       
    for l in f:                                                                 
        if i == 0:                                                              
            i = 1                                                               
            continue                                                            
        gold_map[l.strip()] = True                                              
    f.close()                                                                   
                                                                                
    tp = 0                                                                      
    total_pairs = 0                                                             
                                                                                
    for l in output_pairs:                                               
        total_pairs += 1                                                        
        if gold_map.get(l) is not None:                                         
            tp += 1                                                             
    print tp, total_pairs                                                       
    if tp == 0 or total_pairs == 0:                                             
        pr = 1.0                                                                
        re = 0                                                                  
    else:                                                                       
        pr = float(tp) / float(total_pairs)                                     
        re = float(tp) / float(len(gold_map.keys()))                            
                                                                                
    f1 = (2.0 * pr * re) / (pr + re)                                            
                                                                                
    return (pr, re, f1)     

if __name__ == '__main__':                                                      
    gold_path = sys.argv[1]                                                     
    output_path = sys.argv[2]                                                   
                                                                                
    f = open(output_path, 'r')                                                  
    output_pairs = {}                                                           
    for l in f:                                                                 
        output_pairs[l.strip()] = True                                          
    f.close()                                                                   
                                                                                
    acc = compute_accuracy(gold_path, output_pairs)                             
    print 'precision : ', acc[0]                                                
    print 'recall : ', acc[1]                                                   
    print 'f1 : ', acc[2]     
