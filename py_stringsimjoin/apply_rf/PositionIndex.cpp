#include "PositionIndex.h"

PositionIndex::PositionIndex(std::vector<std::vector<int> >& token_vectors, double& t) {
  int prefix_length, token, i, j, m, n=token_vectors.size(), min_l=100000, max_l=0;
  for(i=0; i<n; i++) {                                                      
    std::vector<int> tokens = token_vectors[i];                                           
    m = tokens.size();                                                   
    size_vector.push_back(m);                                       
    prefix_length = int(m - ceil(threshold * m) + 1.0);                  

    for(j=0; j<prefix_length; j++) {                                      
      index[tokens[j]].push_back(std::pair<int, int>(i, j));           
    }

    if(m > max_len) {                                                     
      max_l = m;
    }                                                     

    if(m < min_len) {                                                     
      min_l = m;
    }                                                     

    threshold = t;                                             
    min_len = min_l;                                                  
    max_len = max_l;
  }  
}

PositionIndex::PositionIndex() {}

PositionIndex::~PositionIndex() {} 
