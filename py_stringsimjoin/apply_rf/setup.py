from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize([
    Extension("tokenizers", sources=["tokenizers.pyx"], language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"]),
    Extension("sim_functions", sources=["sim_functions.pyx"], language="c++",         
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"]),  
    Extension("set_sim_join", sources=["set_sim_join.pyx", "position_index.cpp"], language="c++",             
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"], 
              extra_link_args=['-fopenmp']),
    Extension("overlap_coefficient_join", sources=["overlap_coefficient_join.pyx", "inverted_index.cpp"], language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']),
    Extension("edit_distance_join", sources=["edit_distance_join.pyx", "inverted_index.cpp"], language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']),  
    Extension("executor", sources=["executor.pyx", "node.cpp", "predicatecpp.cpp", "tree.cpp","rule.cpp","coverage.cpp"], language="c++",         
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']),
    Extension("ex_plan", sources=["ex_plan.pyx", "tree.cpp", "rule.cpp", "predicatecpp.cpp", "coverage.cpp", "node.cpp"], language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']),
    Extension("utils", sources=["utils.pyx", "inverted_index.cpp"], language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']),
    Extension("sample", sources=["sample.pyx", "inverted_index.cpp"], language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']),
 ]))
