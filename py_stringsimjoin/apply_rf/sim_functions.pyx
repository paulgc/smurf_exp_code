
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free, abs
from libcpp.vector cimport vector             
from libcpp.set cimport set as oset                                             
                                  

cdef double jaccard(const vector[int]& tokens1, const vector[int]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()           
    cdef int sum_of_size = size1 + size2                                        
#    if sum_of_size == 0:                                                        
#        return 1.0                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                                                        
    while i < size1 and j < size2:                                              
        if tokens1[i] == tokens2[j]:                                            
            overlap += 1                                                        
            i += 1                                                              
            j += 1                                                              
        elif tokens1[i] < tokens2[j]:                                           
            i += 1                                                              
        else:                                                                   
            j += 1                                                              
    return (overlap * 1.0) / <double>(sum_of_size - overlap)       

cdef double dice(const vector[int]& tokens1, const vector[int]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()           
    cdef int sum_of_size = size1 + size2                                        
#    if sum_of_size == 0:                                                        
#        return 1.0                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                                                        
    while i < size1 and j < size2:                                              
        if tokens1[i] == tokens2[j]:                                            
            overlap += 1                                                        
            i += 1                                                              
            j += 1                                                              
        elif tokens1[i] < tokens2[j]:                                           
            i += 1                                                              
        else:                                                                   
            j += 1                                                              
    return (overlap * 2.0) / <double>sum_of_size

cdef double cosine(const vector[int]& tokens1, const vector[int]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()           
#    cdef int sum_of_size = size1 + size2                                        
#    if sum_of_size == 0:                                                        
#        return 1.0                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                                                        
    while i < size1 and j < size2:                                              
        if tokens1[i] == tokens2[j]:                                            
            overlap += 1                                                        
            i += 1                                                              
            j += 1                                                              
        elif tokens1[i] < tokens2[j]:                                           
            i += 1                                                              
        else:                                                                   
            j += 1                                                              
    return <double>overlap / sqrt(size1*size2)

cdef inline int int_min3(int a, int b, int c) nogil:
    if (a<=b) and (a<= c):
        return a
    elif (b<=c):
        return b
    else:
        return c

cdef int int_min(int a, int b) nogil:                                           
    return a if a <= b else b     

cdef double overlap(const vector[int]& tokens1, const vector[int]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                                                        
    while i < size1 and j < size2:                                              
        if tokens1[i] == tokens2[j]:                                            
            overlap += 1                                                        
            i += 1                                                              
            j += 1                                                              
        elif tokens1[i] < tokens2[j]:                                           
            i += 1                                                              
        else:                                                                   
            j += 1                                                              
    return <double>(overlap * 1.0)

cdef double overlap_coefficient(const vector[int]& tokens1, const vector[int]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                                                        
    while i < size1 and j < size2:                                              
        if tokens1[i] == tokens2[j]:                                            
            overlap += 1                                                        
            i += 1                                                              
            j += 1                                                              
        elif tokens1[i] < tokens2[j]:                                           
            i += 1                                                              
        else:                                                                   
            j += 1                                                              
    return (overlap * 1.0) / <double>int_min(size1, size2)          

cdef double left_length(const string& str1, const string& str2) nogil:
    return <double>str1.length()

cdef double right_length(const string& str1, const string& str2) nogil:          
    return <double>str2.length()   

cdef double length_sum(const string& str1, const string& str2) nogil:          
    return <double>(str1.length() + str2.length())  

cdef double length_diff(const string& str1, const string& str2) nogil:           
    return <double>abs(str1.length() - str2.length())   

cdef double edit_distance(const string& str1, const string& str2) nogil:
    cdef int len_str1 = str1.length(), len_str2 = str2.length()

    cdef int ins_cost = 1, del_cost = 1, sub_cost = 1, trans_cost = 1

    cdef int edit_dist, i = 0, j = 0

    if len_str1 == 0:
        return len_str2 * ins_cost

    if len_str2 == 0:
        return len_str1 * del_cost

    cdef int *d_mat = <int*>malloc((len_str1 + 1) * (len_str2 + 1) * sizeof(int)) 
#    cdef int[:,:] d_mat = <int[:(len_str1 + 1), :(len_str2 + 1)]>arr

    for i in range(len_str1 + 1):
        d_mat[i*(len_str2 + 1)] = i * del_cost

    for j in range(len_str2 + 1):
        d_mat[j] = j * ins_cost

    cdef unsigned char lchar = 0
    cdef unsigned char rchar = 0

    for i in range(len_str1):
        lchar = str1[i]
        for j in range(len_str2):
            rchar = str2[j]

            d_mat[(i+1)*(len_str2 + 1) + j+1] = int_min3(d_mat[(i + 1)*(len_str2 + 1) + j] + ins_cost,
                                     d_mat[i*(len_str2 + 1) + j + 1] + del_cost,
                                     d_mat[i*(len_str2 + 1) + j] + (sub_cost if lchar != rchar else 0))
    edit_dist = d_mat[len_str1*(len_str2 + 1) + len_str2]
    free(d_mat)
    return <double>edit_dist

cdef double jaccard_str(vector[string]& tokens1, vector[string]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()           
    cdef int sum_of_size = size1 + size2                                        
    if sum_of_size == 0:                                                        
        return 1.0                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0               
    cdef oset[string] ltokens
    cdef string token
    for token in tokens1:
        ltokens.insert(token)
    for token in tokens2:
        if ltokens.find(token) != ltokens.end():
            overlap += 1                                         
    return (overlap * 1.0) / <double>(sum_of_size - overlap)

cdef double dice_str(vector[string]& tokens1, vector[string]& tokens2) nogil: 
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()           
    cdef int sum_of_size = size1 + size2                                        
    if sum_of_size == 0:                                                        
        return 1.0                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                                                        
    cdef oset[string] ltokens                                                   
    cdef string token                                                           
    for token in tokens1:                                                       
        ltokens.insert(token)                                                   
    for token in tokens2:                                                       
        if ltokens.find(token) != ltokens.end():                                
            overlap += 1                                        
    return (overlap * 2.0) / <double>sum_of_size                                
                                                                                
cdef double cosine_str(vector[string]& tokens1, vector[string]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()           
    cdef int sum_of_size = size1 + size2                                        
    if sum_of_size == 0:                                                        
        return 1.0                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                                                        
    cdef oset[string] ltokens                                                   
    cdef string token                                                           
    for token in tokens1:                                                       
        ltokens.insert(token)                                                   
    for token in tokens2:                                                       
        if ltokens.find(token) != ltokens.end():                                
            overlap += 1                                        
    return <double>overlap / sqrt(size1*size2)                             
