
import py_stringsimjoin as ssj
from py_stringsimjoin.labeler.labeler import *
from py_stringsimjoin.active_learner.active_learner import *
from sklearn.ensemble import RandomForestClassifier                             
from sklearn.linear_model import LogisticRegression
import pandas as pd
import py_stringmatching as sm
from py_stringsimjoin.apply_rf.sample import sample_cython
#from py_stringsimjoin.sampler.sample import *
from py_stringsimjoin.feature.autofeaturegen import *
from py_stringsimjoin.feature.extractfeatures import *
from py_stringsimjoin.apply_rf.apply_rf import *
from py_stringsimjoin.apply_rf.estimate_parameters import *
from py_stringsimjoin.apply_rf.execution_plan import *
from py_stringsimjoin.apply_rf.extract_rules import *
from py_stringsimjoin.utils.tokenizers import *

TABLE_A_PATH = ''
# Keep the ID attr name in both tables to be 'id'
# Make sure the IDs are consecutive integers starting from 0                               
TABLE_A_ID_ATTR = 'id'
TABLE_A_JOIN_ATTR = ''

TABLE_B_PATH = ''
# Keep the ID attr name in both tables to be 'id'
# Make sure the IDs are consecutive integers starting from 0 
TABLE_B_ID_ATTR = 'id'
TABLE_B_JOIN_ATTR = ''

# Seed pairs. E.g., two matches and two non-matches.
SEED_PATH = ''

# Actual matches path.  
GOLD_PATH = ''

OUT_SAMPLE_PATH = ''
OUT_RF_PATH = ''

BATCH_SIZE = 20
MAX_ITERS = 20

ldf=pd.read_csv(TABLE_A_PATH)
rdf=pd.read_csv(TABLE_B_PATH)

seed=pd.read_csv(SEED_PATH)

print ('Performing sampling..')
c=sample_cython(ldf, rdf, TABLE_A_ID_ATTR, TABLE_B_ID_ATTR,
                TABLE_A_JOIN_ATTR, TABLE_B_JOIN_ATTR, 
                100000, 50, seed)
c.to_csv(OUT_SAMPLE_PATH, index=False)

# Note that this simulates the user labeling using the GOLD data
labeled_c = label_table_using_gold(c, 'l_' + TABLE_A_ID_ATTR, 'r_' + TABLE_B_ID_ATTR, GOLD_PATH)
print ('number of positives (after inverted_index sampling) : ', sum(labeled_c['label']))

ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE', 'LEFT_LENGTH', 'RIGHT_LENGTH', 'LENGTH_SUM', 'LENGTH_DIFF'])
#ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE'])

print ('Extracting feature vectors..')
fvs = extract_feature_vecs(c, 'l_' + TABLE_A_ID_ATTR, 'r_' + TABLE_B_ID_ATTR,
                           ldf, rdf, TABLE_A_ID_ATTR, TABLE_B_ID_ATTR,
                           TABLE_A_JOIN_ATTR, TABLE_B_JOIN_ATTR, ft, n_jobs=4)

rf=RandomForestClassifier(n_estimators=10)
al=ActiveLearner(rf, BATCH_SIZE, MAX_ITERS, GOLD_PATH, seed, -1)
lp = al.learn(fvs, '_id', 'l_' + TABLE_A_ID_ATTR, 'r_' + TABLE_B_ID_ATTR)

# Dump random forest to file
import pickle
pickle.dump(al.matcher, open(OUT_RF_PATH, 'w'))                
