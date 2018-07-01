
import subprocess
import numpy as np

def param(cmd):
    cml = "python3 ensembler_bayesian.py " + cmd
    subprocess.call(cml, shell=True)
    ret = np.load("ret.npy")
    return ret[0]

# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)

    a = str(params['NLP_USER'][0])
    b = str(params['TOP_POP'][0])
    c = str(params['NLP_RP3BETA'][0])


    cmd = a + " " + b + " " + c
    ret = param(cmd)
    print("risultato:")
    print(ret)
    return ret