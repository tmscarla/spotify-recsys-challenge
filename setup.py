"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='CreamyFireflies',

    version='1.0.0',

    description='recsys spotify challenge',
    long_description=long_description,

    # The project's main homepage.
    url='polimi.it',

    # Author details
    author='CreamyFireflies',
    author_email='creamy.fireflies@mail.polimi.it',

    # Choose your license
    license='apache-2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[

        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Recommender Systems',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6'
    ],


    keywords=['recommender systems','Spotify','RecSys'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['pandas', 'numpy', 'scipy',
                      'cython', 'tqdm', 'scikit-optimize','spotipy'
                       ],

    extras_require={},

    package_data={},

    data_files=[],

    entry_points={},
)