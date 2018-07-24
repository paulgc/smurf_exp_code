#pragma once
#include "tokenizer.h"

class DelimiterTokenizer : public Tokenizer {
  public:
    string delimiters;
    bool return_set;

    DelimiterTokenizer(string delimiters, bool return_set);
    ~DelimiterTokenizer();

    vector<string> tokenize(const string& str);
};
