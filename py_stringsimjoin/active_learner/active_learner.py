
from math import log
import operator                                                                 
import random

from py_stringsimjoin.sampler.weighted_random_sampler import \
                                                        WeightedRandomSampler


class ActiveLearner:
    def __init__(self, matcher, batch_size, max_iters, gold_file, seed, error_rate=-1):
        self.matcher = matcher
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gold_pairs = load_gold_pairs(gold_file)
        self.seed_pairs = load_seed_pairs(seed)
        self.error_rate = error_rate

    def learn(self, candset, candset_key_attr, candset_l_key_attr, 
              candset_r_key_attr):

        unlabeled_pairs = candset.set_index(candset_key_attr) 
        unlabeled_pairs[candset_l_key_attr] = unlabeled_pairs[candset_l_key_attr].astype(str)
        unlabeled_pairs[candset_r_key_attr] = unlabeled_pairs[candset_r_key_attr].astype(str)

        # find the attributes to be used as features
        feature_attrs = list(unlabeled_pairs.columns)
        feature_attrs.remove(candset_l_key_attr)
        feature_attrs.remove(candset_r_key_attr)
               
        # randomly select first batch of pairs to label 
        first_batch = self._get_first_batch(unlabeled_pairs, candset_l_key_attr,
                                            candset_r_key_attr)
        print first_batch
        # get labels for first batch
        labeled_pairs = self._label_pairs(first_batch, candset_l_key_attr, 
                                          candset_r_key_attr)
        print labeled_pairs
        # remove labeled pairs from the unlabeled pairs
        unlabeled_pairs = unlabeled_pairs.drop(labeled_pairs.index)

        current_iter = 0
        
        while current_iter < self.max_iters:
            # train matcher using the current set of labeled pairs
            self.matcher = self.matcher.fit(labeled_pairs[feature_attrs].values,
                                            labeled_pairs['label'].values)

            # select next batch to label
            print('Selecting next batch...')
            current_batch = self._select_next_batch(unlabeled_pairs, 
                                                    feature_attrs)

            # get labels for current batch
            print('Collecting labels...')
            labeled_current_batch = self._label_pairs(current_batch, 
                                        candset_l_key_attr, candset_r_key_attr)

            # remove labeled pairs from the unlabeled pairs                         
            unlabeled_pairs = unlabeled_pairs.drop(labeled_current_batch.index)     
           
            # append the current batch of labeled pairs to the previous 
            # labeled pairs
            labeled_pairs = labeled_pairs.append(labeled_current_batch)

            current_iter += 1
            print 'Iteration :', current_iter
        return labeled_pairs

    def _select_next_batch(self, unlabeled_pairs, feature_attrs):
        # compute the prediction probabilities for the unlabeled pairs
        probabilities = self.matcher.predict_proba(
                            unlabeled_pairs[feature_attrs].values)

        print 'computing entropy'        
        # compute the entropy for the unlabeled pairs
        entropies = {}
        for i in xrange(len(probabilities)):
            entropy = self._compute_entropy(probabilities[i])
            if entropy > 0:
                entropies[i] = entropy

        print 'sorting'
        # select top k unlabeled pairs based on entropy value.
        top_k_pairs = sorted(entropies.items(),                           
            key=operator.itemgetter(1), reverse=True)[:min(100, len(entropies))]

        print 'sampling'
        next_batch_idxs = []           
        if len(top_k_pairs) <= self.batch_size:
            # if the number of unlabeled pairs whose entropy is above zero is
            # already less than the batch size, then select all of them.
            next_batch_idxs = map(lambda val: val[0], top_k_pairs)
        else:
            # do a weighted random sampling to select the next batch of pairs
            # to be labeled with entropy as the weight.
            weights = map(lambda val: val[1], top_k_pairs)                          
            selected_pairs = map(lambda val: False, top_k_pairs) 
            print weights                                           
            wrs = WeightedRandomSampler(weights)
            while len(next_batch_idxs) < self.batch_size:
                pair_idx = wrs.next()
                if selected_pairs[pair_idx]:
                    continue
                selected_pairs[pair_idx] = True
                next_batch_idxs.append(top_k_pairs[pair_idx][0])

        return unlabeled_pairs.iloc[next_batch_idxs]

    def _compute_entropy(self, arr):
        entropy = 0
        for prob in arr:
            if prob > 0:
                entropy += prob * log(prob)
        if entropy != 0:
            entropy = entropy * -1
        return entropy    

    def _label_pairs(self, to_be_labeled_pairs, l_key_attr, r_key_attr):
        labels = (to_be_labeled_pairs[l_key_attr].astype(str) + ',' + 
            to_be_labeled_pairs[r_key_attr].astype(str)).apply(
                                        lambda val: self._foo(val))
        to_be_labeled_pairs['label'] = labels
        return to_be_labeled_pairs

    def _foo(self, val):
        lab = self.gold_pairs.get(val, 0)
        if random.random() <= self.error_rate:
            return (1 if lab == 0 else 0)
        else:
            return lab

    def _get_first_batch(self, unlabeled_pairs, l_key_attr, r_key_attr):
        
        return unlabeled_pairs[unlabeled_pairs.apply(lambda row: 
            self.seed_pairs.get(str(row[l_key_attr]) + ',' + 
                                str(row[r_key_attr])) != None, 1)].copy()


def load_gold_pairs(gold_file):
    gold_pairs = {}
    file_handle = open(gold_file, 'r')
    for line in file_handle:
        gold_pairs[line.strip()] = 1
    file_handle.close()
    return gold_pairs


def load_seed_pairs(seed):                                                 
    seed_pairs = {}
    for seed_pair_row in seed.itertuples(index=False):
        seed_pairs[str(seed_pair_row[0]) + ',' + 
                   str(seed_pair_row[1])] = int(seed_pair_row[2])                                                        
    return seed_pairs   
