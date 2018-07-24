#include "regex_tokenizer.h"
#include <iostream>

int main(int argc, char* argv[]) {
  RegexTokenizer* rg = new RegexTokenizer("[a-zA-Z]+", true);
  vector<string> tokens = rg->tokenize(argv[1]);
  for(int i=0; i<tokens.size(); i++) {std::cout << tokens[i] << ",";}
  std::cout <<"\n";
} 
