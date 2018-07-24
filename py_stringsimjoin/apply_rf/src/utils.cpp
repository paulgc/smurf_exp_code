#include "utils.h"

#include <unordered_set>

vector<string> remove_duplicates(const vector<string>& inp_vector) {
  vector<string> out_tokens;
  std::unordered_set<string> seen_tokens;
  for (vector<string>::const_iterator it = inp_vector.begin(); it != inp_vector.end(); ++it) {
    if (seen_tokens.find(*it) == seen_tokens.end()) {
      out_tokens.push_back(*it);
      seen_tokens.insert(*it);
    }
  }
  return out_tokens;
}
