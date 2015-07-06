from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

setup(name = 'HMM',
       ext_modules = cythonize(Extension(
           "hmm",                                # the extesion name
           sources=["hmm.pyx",
			os.sep.join([".", "src", "base", "MHMMEM.cpp"]),
			os.sep.join([".", "src", "base", "ESSMHMM.cpp"]),
			os.sep.join([".", "src", "base", "FwdBack.cpp"]),
			os.sep.join([".", "src", "io",  "Util.cpp"]),
			os.sep.join([".", "src", "base", "MatlabUtils.cpp"]),
			os.sep.join([".", "src", "base", "MixGaussProb.cpp"]),
			os.sep.join([".", "src", "base", "MixgaussMstep.cpp"]),
			os.sep.join([".", "src", "base", "GMM.cpp"]),
			os.sep.join([".", "src", "base", "KMeansInit.cpp"]),
			os.sep.join([".", "src", "base", "Util.cpp"]),
			os.sep.join([".", "src", "base", "MixGaussInit.cpp"])
                    ], # the Cython source and
                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
           extra_compile_args=["-I " + os.sep.join([".", "eigen3", ""])],
           include_dirs = [numpy.get_include(), os.sep.join([".", "eigen3", ""])]

      )))
