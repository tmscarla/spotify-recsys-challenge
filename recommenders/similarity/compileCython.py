#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    author: Simone Boglio
    mail: simone.boglio@mail.polimi.it
"""

# how call this script:
# python compileCython.py build_ext --inplace


try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


from Cython.Distutils import build_ext
import numpy
import sys
import re
import glob
import logging
import os.path
import platform

from setuptools import Extension, setup, find_packages


#files_to_compile = ['tversky'] # for testing or recompile just one 

files_to_compile = ['cosine','dot_product','jaccard','tversky','dice','p3alpha_rp3beta','s_plus']


try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

if not use_cython:
    raise RuntimeError('Cython required to build files')

use_openmp = True

def define_extensions(use_cython=False):
    if sys.platform.startswith("win"):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        compile_args = ['/O2', '/openmp']
        link_args = []
    else:
        gcc = extract_gcc_binaries()
        if gcc is not None:
            rpath = '/usr/local/opt/gcc/lib/gcc/' + gcc[-1] + '/'
            link_args = ['-Wl,-rpath,' + rpath]
        else:
            link_args = []

        compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-O3', '-ffast-math']
        if use_openmp:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")

        compile_args.append("-std=c++11")
        link_args.append("-std=c++11")

    src_ext = '.pyx' if use_cython else '.cpp'
    modules = [Extension(name,
                         [name + src_ext],
                         language='c++',
                         extra_compile_args=compile_args, extra_link_args=link_args)
               for name in files_to_compile]

    if use_cython:
        return cythonize(modules)
    else:
        return modules


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python

def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = ['/opt/local/bin/g++-mp-[0-9].[0-9]',
                '/opt/local/bin/g++-mp-[0-9]',
                '/usr/local/bin/g++-[0-9].[0-9]',
                '/usr/local/bin/g++-[0-9]']
    if 'darwin' in platform.platform().lower():
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None


def set_gcc():
    """Try to use GCC on OSX for OpenMP support."""
    # For macports and homebrew

    if 'darwin' in platform.platform().lower():
        gcc = extract_gcc_binaries()

        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')


set_gcc()


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=define_extensions(use_cython)
)








