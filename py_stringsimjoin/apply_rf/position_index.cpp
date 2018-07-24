#include "position_index.h"

PositionIndex::PositionIndex(std::map<int, std::vector< std::pair<int, int> > >& ind, std::vector<int>& sv, int& min_l, int& max_l, double& t) {
  index = ind;
  size_vector = sv;
  min_len = min_l;
  max_len = max_l;
  threshold = t;
}

void PositionIndex::set_fields(std::map<int, std::vector< std::pair<int, int> > >& ind, std::vector<int>& sv, int& min_l, int& max_l, double& t) {
  index = ind;                                                                  
  size_vector = sv;                                                             
  min_len = min_l;                                                              
  max_len = max_l;                                                              
  threshold = t;                                                                
}

PositionIndex::PositionIndex() {}

PositionIndex::~PositionIndex() {} 
