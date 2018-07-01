
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
    a = str(params['CBF_ALBUM'][0])
    b = str(params['CBF_ARTISTA'][0])
    c = str(params['NLP_USER'][0])
    d = str(params['RP3BETA'][0])
    e = str(params['CF_USER'][0])
    f = str(params['SLIM'][0])
    g = str(params['CBF_USER_ARTIST'][0])
    cmd = a + " " + b + " " + c + " " + d + " " + e + " " + f + " " + g
    ret = param(cmd)
    print("risultato:")
    print(ret)
    return ret