#pragma once
#include "tokenizer.h"

class QgramTokenizer : public Tokenizer {
  public:
    int qval;
    char prefix_pad;
    char suffix_pad;
    bool padding;
    bool return_set;

    QgramTokenizer(int qval, bool padding, char prefix_pad, char suffix_pad, 
                   bool return_set);
    ~QgramTokenizer();

    vector<string> tokenize(const string& str);
};
