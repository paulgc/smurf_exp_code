#include "delimiter_tokenizer.h"
#include "utils.h"

#include <set>

DelimiterTokenizer::DelimiterTokenizer(
  string delimiters, bool return_set): delimiters(delimiters), return_set(return_set) {
}

DelimiterTokenizer::~DelimiterTokenizer() {}

vector<string> DelimiterTokenizer::tokenize(const string& str) {
  vector<string> tokens;
  string::size_type last_pos = str.find_first_not_of(delimiters, 0);
  string::size_type pos = str.find_first_of(delimiters, last_pos);
  while (string::npos != pos || string::npos != last_pos) {
    string token = str.substr(last_pos, pos - last_pos);
    tokens.push_back(token);
    last_pos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, last_pos);
  }
  return return_set ? remove_duplicates(tokens) : tokens;
}
