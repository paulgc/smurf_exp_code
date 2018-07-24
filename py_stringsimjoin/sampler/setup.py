from distutils.core import setup, Extension                                     
from Cython.Build import cythonize                                              
                                                                                
setup(ext_modules = cythonize([                                                 
    Extension("sample", sources=["sample.pyx", "inverted_index.cpp"], language="c++",                   
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp'], include_dirs=["../apply_rf/"]),                                    
 ]))   
