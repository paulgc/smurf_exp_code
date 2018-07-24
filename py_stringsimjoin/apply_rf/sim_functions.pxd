
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef double jaccard(const vector[int]& tokens1, const vector[int]& tokens2) nogil

cdef double dice(const vector[int]& tokens1, const vector[int]& tokens2) nogil

cdef double cosine(const vector[int]& tokens1, const vector[int]& tokens2) nogil

cdef double overlap(const vector[int]& tokens1, const vector[int]& tokens2) nogil

cdef double overlap_coefficient(const vector[int]& tokens1, const vector[int]& tokens2) nogil

cdef double edit_distance(const string& str1, const string& str2) nogil

cdef double jaccard_str(vector[string]& tokens1, vector[string]& tokens2) nogil
                                                                                
cdef double dice_str(vector[string]& tokens1, vector[string]& tokens2) nogil  
                                                                                
cdef double cosine_str(vector[string]& tokens1, vector[string]& tokens2) nogil

cdef double left_length(const string& str1, const string& str2) nogil
cdef double right_length(const string& str1, const string& str2) nogil
cdef double length_sum(const string& str1, const string& str2) nogil
cdef double length_diff(const string& str1, const string& str2) nogil
