#include "inverted_index.h"

InvertedIndex::InvertedIndex(std::map<int, std::vector<int> >& ind, std::vector<int>& sv) {
  index = ind;
  size_vector = sv;
}

void InvertedIndex::set_fields(std::map<int, std::vector<int> >& ind, std::vector<int>& sv) {
  index = ind;                                                                  
  size_vector = sv;                                                             
}

InvertedIndex::InvertedIndex() {}

InvertedIndex::~InvertedIndex() {} 
