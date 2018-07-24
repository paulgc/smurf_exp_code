#include "qgram_tokenizer.h"
#include "utils.h"

#include <set>

QgramTokenizer::QgramTokenizer(int qval, bool padding, char prefix_pad, 
  char suffix_pad, bool return_set): qval(qval), padding(padding), prefix_pad(prefix_pad), suffix_pad(suffix_pad), return_set(return_set) {
}

QgramTokenizer::~QgramTokenizer() {}

vector<string> QgramTokenizer::tokenize(const string& str) {
  string inp_str = str;
  if (padding) {
    inp_str = string(qval - 1, prefix_pad) + inp_str + string(qval - 1, suffix_pad);
  }
  vector<string> tokens;
  for (int i = 0; i <= (inp_str.length() - qval); ++i) {
    tokens.push_back(inp_str.substr(i, qval));
  }
  return return_set ? remove_duplicates(tokens) : tokens;
}
