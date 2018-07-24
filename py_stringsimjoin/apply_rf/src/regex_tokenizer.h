#pragma once
#include "tokenizer.h"

#include <regex>

class RegexTokenizer : public Tokenizer {
  public:
    std::regex token_regex;
    bool return_set;

    RegexTokenizer(string regex_str, bool return_set);
    ~RegexTokenizer();

    vector<string> tokenize(const string& str);
};
