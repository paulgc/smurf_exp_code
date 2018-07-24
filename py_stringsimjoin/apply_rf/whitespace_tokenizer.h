#pragma once
#include "tokenizer.h"

class WhitespaceTokenizer : public Tokenizer {
  public:
    bool return_set;

    WhitespaceTokenizer(bool return_set);
    WhitespaceTokenizer();
    ~WhitespaceTokenizer();

    vector<string> tokenize(const string& str);
};
