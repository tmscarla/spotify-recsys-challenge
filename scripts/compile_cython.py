import os
from recommenders.similarity.definitions import ROOT_DIR, tail
filepath = os.path.join(ROOT_DIR,tail)

os.chdir(filepath)
try:
    os.rename('__init__.py','__init__2.py')
    rename=True
except:
    rename=False
try:
    os.system('python compileCython.py build_ext --inplace')
    print('COMPILING CYTHON DONE')
except:
    print('ERROR IN COMPILING CYTHON')
if rename:
    try:  
        os.rename('__init__2.py','__init__.py')
    except:
        print('ERROR: raneming __init__.py file')



