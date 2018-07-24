#include "coverage.h"

Coverage::Coverage(std::vector<bool>& cov) {
  int i=0;
  count = 0;
  size = cov.size();
  for (std::vector<bool>::iterator it=cov.begin(); it!=cov.end(); ++it) {
    if (*it) {
      bit_vector.flip(i);
      count += 1;
    }
    i += 1;
  }
}

Coverage::Coverage() {}

Coverage::~Coverage() {}

int Coverage::and_sum(const Coverage& cov) {
  return (bit_vector & cov.bit_vector).count();
}

void Coverage::or_coverage(const Coverage& cov) {
  bit_vector |= cov.bit_vector;
}

void Coverage::and_coverage(const Coverage& cov) {                               
  bit_vector &= cov.bit_vector;                                                 
}

void Coverage::reset() {
  bit_vector.reset();
}

int Coverage::sum() {
  return bit_vector.count();
}  
