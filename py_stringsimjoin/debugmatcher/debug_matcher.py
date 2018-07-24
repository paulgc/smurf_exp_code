
import pandas as pd

def debug_matcher(matcher, train, test, feat_attr, target_attr,
                  fk_ltable, fk_rtable, ltable, rtable, 
                  l_key_attr, r_key_attr):
    matcher.fit(train[[feat_attr]].values, train[target_attr].values)
    threshold = -float(matcher.intercept_)/float(matcher.coef_)
    print threshold
    false_pos = test[test.apply(lambda row: row[feat_attr] >= threshold and 
                                            row[target_attr] == 0, axis=1)]
    false_neg = test[test.apply(lambda row: row[feat_attr] < threshold and     
                                            row[target_attr] == 1, axis=1)]

    indexed_ltable = ltable.set_index(l_key_attr)
    indexed_rtable = rtable.set_index(r_key_attr)

    ltable_cols = map(lambda col_name: 'l_'+col_name, 
                      list(indexed_ltable.columns))
    rtable_cols = map(lambda col_name: 'r_'+col_name,                           
                      list(indexed_rtable.columns))  

    false_pos_ltable_rows = []
    false_pos_rtable_rows = []
    for idx, row in false_pos.iterrows():
        false_pos_ltable_rows.append(indexed_ltable.ix[row[fk_ltable]].values)
        false_pos_rtable_rows.append(indexed_rtable.ix[row[fk_rtable]].values) 

    false_pos_l_df = pd.DataFrame(false_pos_ltable_rows, index=false_pos.index.values, columns=ltable_cols)
    false_pos_r_df = pd.DataFrame(false_pos_rtable_rows, index=false_pos.index.values, columns=rtable_cols)   
    false_pos_df = pd.concat([false_pos, false_pos_l_df, false_pos_r_df], 
                             axis=1)

    false_neg_ltable_rows = []                                                  
    false_neg_rtable_rows = []                                                  
    for idx, row in false_neg.iterrows():                                       
        false_neg_ltable_rows.append(indexed_ltable.ix[row[fk_ltable]].values)
        false_neg_rtable_rows.append(indexed_rtable.ix[row[fk_rtable]].values)

    false_neg_l_df = pd.DataFrame(false_neg_ltable_rows, index=false_neg.index.values, columns=ltable_cols)   
    false_neg_r_df = pd.DataFrame(false_neg_rtable_rows, index=false_neg.index.values, columns=rtable_cols)   
    false_neg_df = pd.concat([false_neg, false_neg_l_df, false_neg_r_df],       
                             axis=1)

    predicted_pos = len(test[test[feat_attr] >= threshold])
    precision = float(predicted_pos - len(false_pos)) / float(predicted_pos)
    recall = float(predicted_pos - len(false_pos)) / float(len(test[test[target_attr] == 1]))
    f1 = (2.0 * precision * recall) / float(precision + recall)

    stats = {'precision': precision, 'recall':recall, 'f1':f1}
    return {'stats' : stats, 'false_pos' : false_pos_df, 'false_neg' : false_neg_df}

def what_if_matrix(sample, feat_attr, target_attr, threshold, delta, n):
    threshold_matrix = []
    curr_t = max(float(threshold) - float(delta * n), 0)
    end = min(float(threshold) + float(delta * n), 1)
    while curr_t <= end:
        predicted_pos = sample[sample[feat_attr] >= curr_t]
        false_pos = predicted_pos[predicted_pos[target_attr] == 0]
        precision = float(len(predicted_pos) - len(false_pos)) / float(len(predicted_pos))    
        recall = float(len(predicted_pos) - len(false_pos)) / float(len(sample[sample[target_attr] == 1]))
        f1 = (2.0 * precision * recall) / float(precision + recall)                 
        threshold_matrix.append([curr_t, precision, recall, f1])
        curr_t += delta 
    threshold_matrix_df = pd.DataFrame(threshold_matrix, columns=['threshold','precision','recall','f1'])
    return threshold_matrix_df
