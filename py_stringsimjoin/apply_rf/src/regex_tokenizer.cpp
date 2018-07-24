#include "regex_tokenizer.h"
#include "utils.h"

#include <set>

RegexTokenizer::RegexTokenizer(string regex_str, bool return_set_flag) {
  token_regex = std::regex(regex_str);
  return_set = return_set_flag;
}

RegexTokenizer::~RegexTokenizer() {}

vector<string> RegexTokenizer::tokenize(const string& str) {
  vector<string> tokens;
  std::regex_token_iterator<string::const_iterator> it (str.begin(), str.end(), token_regex);

  std::regex_token_iterator<string::const_iterator> rend;
  while (it != rend) {
    tokens.push_back(*it++);
  }
  return return_set ? remove_duplicates(tokens) : tokens;
}
