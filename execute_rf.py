import pandas as pd
from py_stringsimjoin.apply_rf.executor import *
import pickle
from py_stringsimjoin.feature.autofeaturegen import get_features

# RF produced by learn_rf.py
RF_PATH = ''

# Directory where intermediate files and output will be written
# Output matches will be written to WORKING_DIR in files prefixed with 'output_'
WORKING_DIR = ''                                               

# Path to the sample obtained when running learn_rf.py
SAMPLE_PATH = ''

TABLE_A_PATH = ''
TABLE_B_PATH = ''

TABLE_A_JOIN_ATTR = ''
TABLE_B_JOIN_ATTR = ''

def load_sample(sample_size):
    c = pd.read_csv(SAMPLE_PATH)                                           
    sample = c.sample(sample_size)                                                        
    l_id = []                                                                       
    r_id = []                                                                       
                                                                                
    for idx, row in sample.iterrows():                                              
        l_id.append(row['l_id'])                                                    
        r_id.append(row['r_id']) 

    return (l_id, r_id)

rf=pickle.load(open(RF_PATH, 'r'))
(l1, l2) = load_sample(10000)


ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE', 'LEFT_LENGTH', 'RIGHT_LENGTH', 'LENGTH_SUM', 'LENGTH_DIFF'])
#ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE'])

test_execute_rf(rf, ft, l1, l2, TABLE_A_PATH, TABLE_A_JOIN_ATTR, 
                TABLE_B_PATH, TABLE_B_JOIN_ATTR, WORKING_DIR, 4)
