
import subprocess
import numpy as np


def param(norm,work,split,skip,date,porter,porter2,lanca,lanca2,data1):
    cml = "python3 NLP_main.py"
    cml += " " + str(norm)
    cml += " " + str(work)
    cml += " " + str(split)
    cml += " " + str(skip)
    cml += " " + str(date)
    cml += " " + str(porter)
    cml += " " + str(porter2)
    cml += " " + str(lanca)
    cml += " " + str(lanca2)
    cml += " " + str(data1)
    subprocess.call(cml, shell=True)
    ret = np.load("ret.npy")
    return ret[0]


# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)

    norm = params['norm'][0]
    work = params['work'][0]
    split = params['split'][0]
    skip = params['skip'][0]
    date = params['date'][0]
    porter = params['porter'][0]
    porter2 = params['porter2'][0]
    lanca = params['lanca'][0]
    lanca2 = params['lanca2'][0]
    data1 = params['data1'][0]

    ret = param(norm,work,split,skip,date,porter,porter2,lanca,lanca2,data1)
    print("risultato:")
    print(ret)
    return ret