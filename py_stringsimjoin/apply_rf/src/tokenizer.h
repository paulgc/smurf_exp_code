#pragma once
#include <string>
#include <vector>

using std::string;
using std::vector;

class Tokenizer {
  public:
    Tokenizer();
    virtual ~Tokenizer();

    virtual vector<string> tokenize(const string& str) = 0;
};
