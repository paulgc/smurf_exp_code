
import time
from operator import itemgetter 

from cython.parallel import prange

from libc.stdlib cimport atoi
from libcpp.vector cimport vector
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair                                                   
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE, sprintf           

import re2

cdef extern from "string.h" nogil:                                                    
    char *strtok (char *inp_str, const char *delimiters)  

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort(vector[int].iterator, vector[int].iterator)
 
#cdef extern from "whitespace_tokenizer.h" nogil:                                      
#    cdef cppclass WhitespaceTokenizer nogil:                                          
#        WhitespaceTokenizer()                                                         
#        WhitespaceTokenizer(bool return_set)
#        bool return_set
#        vector[string] tokenize(const string&)
   

cdef class WhitespaceTokenizer:
    cdef bool return_set

    def __init__(self, bool return_set):
        self.return_set = return_set

    cpdef vector[string] tokenize(self, const string& inp_string):
        cdef char* pch                                                              
        pch = strtok (<char*> inp_string.c_str(), " ") 
        cdef oset[string] tokens                                            
        cdef vector[string] out_tokens                                              
        if self.return_set:                             
            while pch != NULL:                                                          
                tokens.insert(string(pch))                                          
                pch = strtok (NULL, " ")
            for s in tokens:
                out_tokens.push_back(s)                                                
        else:
            while pch != NULL:                                                  
                out_tokens.push_back(string(pch))                                      
                pch = strtok (NULL, " ")                                        
        return out_tokens 

cdef class QgramTokenizer:
    cdef int qval                                                               
    cdef char prefix_pad, suffix_pad                                            
    cdef bool padding, return_set        

    def __init__(self, int qval, bool padding, char prefix_pad, char suffix_pad, 
                 bool return_set):                                        
        self.qval = qval
        self.padding = padding
        self.prefix_pad = prefix_pad
        self.suffix_pad = suffix_pad
        self.return_set = return_set                                            
                                                                                
    cpdef vector[string] tokenize(self, const string& inp_string):               
        cdef string inp_str = inp_string;                                                         
        if self.padding:                                                                
            inp_str = string(self.qval - 1, self.prefix_pad) + inp_str + string(self.qval - 1, self.suffix_pad)
        cdef oset[string] tokens                                                
        cdef vector[string] out_tokens     
        cdef unsigned int i, n = inp_str.length() - self.qval + 1
        if self.return_set:
            for i in range(n):                                                  
                tokens.insert(inp_str.substr(i, self.qval))
            for s in tokens:
                out_tokens.push_back(s) 
        else:
            for i in range(n):
                out_tokens.push_back(inp_str.substr(i, self.qval))
        return out_tokens

class AlphabeticTokenizer:                                                                                                          
    def __init__(self, return_set):                                        
        self.regex = re2.compile('[a-zA-Z]+')
        self.return_set = return_set                                            
                                                                                
    def tokenize(self, const string& inp_string):               
        cdef oset[string] tokens                                                
        cdef vector[string] out_tokens                                          
        if self.return_set:
            for s in self.regex.findall(inp_string):                                                     
                tokens.insert(s)                                      
            for s in tokens:                                                    
                out_tokens.push_back(s)                                         
        else:                                                                   
            for s in self.regex.findall(inp_string):                            
                out_tokens.push_back(s)  
        return out_tokens 

class AlphanumericTokenizer:                                                      
    def __init__(self, return_set):                                             
        self.regex = re2.compile('[a-zA-Z0-9]+')                                   
        self.return_set = return_set                                            
                                                                                
    def tokenize(self, const string& inp_string):                               
        cdef oset[string] tokens                                                
        cdef vector[string] out_tokens                                          
        if self.return_set:                                                     
            for s in self.regex.findall(inp_string):                            
                tokens.insert(s)                                                
            for s in tokens:                                                    
                out_tokens.push_back(s)                                         
        else:                                                                   
            for s in self.regex.findall(inp_string):                            
                out_tokens.push_back(s)                                         
        return out_tokens  

class NumericTokenizer:                                                      
    def __init__(self, return_set):                                             
        self.regex = re2.compile('[0-9]+')                                   
        self.return_set = return_set                                            
                                                                                
    def tokenize(self, const string& inp_string):                               
        cdef oset[string] tokens                                                
        cdef vector[string] out_tokens                                          
        if self.return_set:                                                     
            for s in self.regex.findall(inp_string):                            
                tokens.insert(s)                                                
            for s in tokens:                                                    
                out_tokens.push_back(s)                                         
        else:                                                                   
            for s in self.regex.findall(inp_string):                            
                out_tokens.push_back(s)                                         
        return out_tokens  

def test_tok(df, attr):
    cdef int q=3
    ws = QgramTokenizer(q, True, ord('#'), ord('$'), True)
#    ws = AlphanumericTokenizer(True)
    cdef vector[string] strings
    convert_to_vector(df[attr], strings)                           
    cdef vector[string] t
    for s in strings:
        t = ws.tokenize(s)                
        print t                

def test_tok2(df1, attr1, df2, attr2):
    cdef vector[string] lstrings, rstrings                                      
    convert_to_vector(df1[attr1], lstrings)                                    
    convert_to_vector(df2[attr2], rstrings)                                    
    st = time.time()
    tokenize(lstrings, rstrings, 'ws', 'gh1') 
    print 'time : ', time.time() - st

cdef vector[int] split(string inp_string):
    cdef char* pch                                                          
    pch = strtok (<char*> inp_string.c_str(), " ")                          
    cdef vector[int] out_tokens                                          
    while pch != NULL:                                                  
        out_tokens.push_back(atoi(pch))                               
        pch = strtok (NULL, " ")                                        
    return out_tokens  

cpdef void load_tok(tok_type, path, vector[vector[int]] &ltokens, vector[vector[int]] &rtokens):
    st =time.time()                                                             
    fp = open(path+"/ltable_"+tok_type)                                         
    for line in fp:                                                             
        ltokens.push_back(split(line))                                          
    fp.close()                                                                  
    fp = open(path+"/rtable_"+tok_type)                                         
    for line in fp:                                                             
        rtokens.push_back(split(line))                                          
    fp.close() 

cdef bool mycomp(pair[string, int] i, pair[string, int] j):
    return (i.second < j.second)

cdef vector[string] tokenize_str(string& inp_str, const string& tok_type):          
    cdef object tok                                                             
    if tok_type.compare('ws') == 0:                                             
        tok = WhitespaceTokenizer(True)                                         
    elif tok_type.compare('alph') == 0:                                         
        tok = AlphabeticTokenizer(True)                                         
    elif tok_type.compare('alph_num') == 0:                                     
        tok = AlphanumericTokenizer(True)                                       
    elif tok_type.compare('num') == 0:                                          
        tok = NumericTokenizer(True)                                            
    elif tok_type.compare('qg2') == 0:                                          
        tok = QgramTokenizer(2, True, ord('#'), ord('$'), True)                
    elif tok_type.compare('qg3') == 0:                                          
        tok = QgramTokenizer(3, True, ord('#'), ord('$'), True)
    elif tok_type.compare('qg2_bag') == 0:                                          
        tok = QgramTokenizer(2, True, ord('#'), ord('$'), False)          
    return tok.tokenize(inp_str)

cpdef void tokenize(vector[string]& lstrings, vector[string]& rstrings,          
                   const string& tok_type, const string& working_dir):          
    cdef object tok                                                               
    if tok_type.compare('ws') == 0:                                             
        tok = WhitespaceTokenizer(True)                              
    elif tok_type.compare('alph') == 0:                                         
        tok = AlphabeticTokenizer(True)                              
    elif tok_type.compare('alph_num') == 0:                                     
        tok = AlphanumericTokenizer(True)                            
    elif tok_type.compare('num') == 0:                                          
        tok = NumericTokenizer(True)                                 
    elif tok_type.compare('qg2') == 0:                                          
        tok = QgramTokenizer(2, True, ord('#'), ord('$'), True)      
    elif tok_type.compare('qg3') == 0:                                          
        tok = QgramTokenizer(3, True, ord('#'), ord('$'), True)      
    elif tok_type.compare('qg2_bag') == 0:                                      
        tok = QgramTokenizer(2, True, ord('#'), ord('$'), False)  
    print tok_type
    #cdef AlphabeticTokenizer tok
#    tok = AlphabeticTokenizer(True)                                                                                
    cdef string s, token                                                        
    cdef vector[string] tokens                                                  
    cdef omap[string, int] token_freq, token_ordering                           
    cdef vector[vector[string]] ltokens, rtokens                                
    cdef int j, n=lstrings.size()

    for j in range(n):                                                          
        tokens = tok.tokenize(lstrings[j])                                                         
        ltokens.push_back(tokens)                                               
        for token in tokens:                                                    
            token_freq[token] += 1                                              

    n = rstrings.size()                                                 
    for j in range(n):                                                          
        tokens = tok.tokenize(rstrings[j])                                                
        rtokens.push_back(tokens)                                               
        for token in tokens:                                                    
            token_freq[token] += 1                                              

    ordered_tokens = []                              
    for entry in token_freq:
        ordered_tokens.append((entry.first, entry.second))                                                    

    cdef int order_idx = 1
    for token_freq_tuple in sorted(ordered_tokens, key=itemgetter(1)):
        token_ordering[token_freq_tuple[0]] = order_idx
        order_idx += 1

    fp = open(working_dir + "/ltable_" + tok_type, 'w')
    cdef char buf[10]
    cdef string space = " "
    for tokens in ltokens:
        otokens = []
        n = tokens.size()
        for j in range(n):
            otokens.append(token_ordering[tokens[j]])
        otokens.sort()
#        s = ""
#        n = tokens.size() - 1
#        for j in range(n):
#            sprintf(buf, '%d', token_ordering[tokens[j]])
#            s += string(buf) + space
#        sprintf(buf, '%d', token_ordering[tokens[n]])                       
#        s += string(buf)
        fp.write(' '.join(map(str, otokens)) + '\n')
    fp.close()

    fp = open(working_dir + "/rtable_" + tok_type, 'w')                         
    for tokens in rtokens:                                                      
        otokens = []                                                            
        n = tokens.size()                                                       
        for j in range(n):                                                      
            otokens.append(token_ordering[tokens[j]])                           
        otokens.sort()         
        fp.write(' '.join(map(str, otokens)) + '\n')                            
    fp.close()      

cdef void convert_to_vector(string_col, vector[string]& string_vector):         
    for val in string_col:                                                      
        string_vector.push_back(val)

cdef vector[string] remove_duplicates(vector[string]& inp_vector):          
    cdef vector[string] out_tokens                                                    
    cdef oset[string] seen_tokens
    cdef string inp_str                                     
    for inp_str in inp_vector:
        if seen_tokens.find(inp_str) == seen_tokens.end():        
            out_tokens.push_back(inp_str)                                                
            seen_tokens.insert(inp_str)                                                                                                                       
    return out_tokens

cdef void tokenize_without_materializing(vector[string]& lstrings, 
                                          vector[string]& rstrings,         
                                          const string& tok_type,
                                          vector[vector[int]]& l_ordered_tokens,
                                          vector[vector[int]]& r_ordered_tokens):          
    cdef object tok                                                             
    if tok_type.compare('ws') == 0:                                             
        tok = WhitespaceTokenizer(True)                                         
    elif tok_type.compare('alph') == 0:                                         
        tok = AlphabeticTokenizer(True)                                         
    elif tok_type.compare('alph_num') == 0:                                     
        tok = AlphanumericTokenizer(True)                                       
    elif tok_type.compare('num') == 0:                                          
        tok = NumericTokenizer(True)                                            
    elif tok_type.compare('qg2') == 0:                                          
        tok = QgramTokenizer(2, True, ord('#'), ord('$'), True)                
    elif tok_type.compare('qg3') == 0:                                          
        tok = QgramTokenizer(3, True, ord('#'), ord('$'), True)                 
    elif tok_type.compare('qg2_bag') == 0:                                      
        tok = QgramTokenizer(2, True, ord('#'), ord('$'), False)  
                                                                                
    #cdef AlphabeticTokenizer tok                                               
#    tok = AlphabeticTokenizer(True)                                                                                
    cdef string s, token                                                        
    cdef vector[string] tokens                                                  
    cdef omap[string, int] token_freq, token_ordering                           
    cdef vector[vector[string]] ltokens, rtokens                                
    cdef int j, n=lstrings.size()                                               
                                                                                
    for j in range(n):                                                          
        tokens = tok.tokenize(lstrings[j])                                      
        ltokens.push_back(tokens)                                               
        for token in tokens:                                                    
            token_freq[token] += 1                                              
                                                                                
    n = rstrings.size()                                                         
    for j in range(n):                                                          
        tokens = tok.tokenize(rstrings[j])                                      
        rtokens.push_back(tokens)                                               
        for token in tokens:                                                    
            token_freq[token] += 1                                              
                                                                                
    ordered_tokens = []                                                         
    for entry in token_freq:                                                    
        ordered_tokens.append((entry.first, entry.second))                      
                                                                                
    cdef int order_idx = 1                                                      
    for token_freq_tuple in sorted(ordered_tokens, key=itemgetter(1)):          
        token_ordering[token_freq_tuple[0]] = order_idx                         
        order_idx += 1                                                          
   
    cdef vector[int] otokens                                                                             
    for tokens in ltokens:                                                                                                                  
        n = tokens.size()                                                       
        for j in range(n):                                                      
            otokens.push_back(token_ordering[tokens[j]])                           
        sort(otokens.begin(), otokens.end())
        l_ordered_tokens.push_back(otokens)
        otokens.clear()

    for tokens in rtokens:                                                                                           
        n = tokens.size()                                                       
        for j in range(n):                                                      
            otokens.push_back(token_ordering[tokens[j]])                           
        sort(otokens.begin(), otokens.end())
        r_ordered_tokens.push_back(otokens)
        otokens.clear()                        
