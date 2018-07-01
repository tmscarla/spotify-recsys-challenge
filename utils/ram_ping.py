import sys
import psutil
from time import gmtime, strftime, sleep

logFile = open('../results/RAM_USAGE.txt','w')

best = [0,""]
while(True):
    for i in range(60):
        ram = psutil.virtual_memory()
        if ram[3]>best[0]:
            best= [ram[3] , strftime("%Y-%m-%d %H:%M:%S", gmtime())]
        logFile.write(strftime("%Y-%m-%d %H:%M:%S", gmtime())+" "+str(psutil.virtual_memory()[2]+" top: "+str(best[0])+" "+str(best[1]))+"\n")
        logFile.flush()
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime())+" "+str(psutil.virtual_memory()[2]+" top: "+str(best[0])+" "+str(best[1]))+"\n",flush=True)
        sleep(60)