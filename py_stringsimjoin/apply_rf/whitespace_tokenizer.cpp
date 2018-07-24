#include "whitespace_tokenizer.h"

#include <set>                                                        

WhitespaceTokenizer::WhitespaceTokenizer(bool return_set): return_set(return_set) {
}

WhitespaceTokenizer::~WhitespaceTokenizer() {}

WhitespaceTokenizer::WhitespaceTokenizer() {}                                  

vector<string> WhitespaceTokenizer::tokenize(const string& str) {
  vector<string> tokens;
  string::size_type last_pos = str.find_first_not_of(" ", 0);
  string::size_type pos = str.find_first_of(" ", last_pos);
  if (return_set) {
    std::set<string> seen_tokens;                                       
    while (string::npos != pos || string::npos != last_pos) {
      string token = str.substr(last_pos, pos - last_pos);
      seen_tokens.insert(token);
      last_pos = str.find_first_not_of(" ", pos);
      pos = str.find_first_of(" ", last_pos);
    }
    for (std::set<string>::iterator it=seen_tokens.begin(); it!=seen_tokens.end(); ++it) {
      tokens.push_back(*it);
    } 
    return tokens;
  } else {
    while (string::npos != pos || string::npos != last_pos) {                   
      string token = str.substr(last_pos, pos - last_pos);                      
      tokens.push_back(token);                                                  
      last_pos = str.find_first_not_of(" ", pos);                        
      pos = str.find_first_of(" ", last_pos);                            
    }  
    return tokens;
  }
}
