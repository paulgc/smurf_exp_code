#pragma once
#include <vector>
#include <string>

using std::vector;
using std::string;

enum similarity_measure_type {                                                  
  COSINE,                                                                       
  DICE,                                                                         
  EDIT_DISTANCE,                                                                
  JACCARD,                                                                      
  OVERLAP,                                                                      
  OVERLAP_COEFFICIENT                                                           
};                                                                              
                                                                                
enum comparison_operator {                                                      
  EQ,                                                                           
  GE,                                                                           
  GT,                                                                           
  LE,                                                                           
  LT,                                                                           
  NE                                                                            
}; 

vector<string> remove_duplicates(const vector<string>& inp_vector);
