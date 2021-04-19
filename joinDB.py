
import pandas as pd
import numpy as np



cadena="MOEAD_DTLZ1_03D_R"
type1=".pos"
type2=".pof"
nameFile=str()

#x=np.loadtxt("MOEAD_DTLZ1_03D_R01.pof")
#x1=pd.read_table("MOEAD_DTLZ1_03D_R01.pof")
x2=open("MOEAD_DTLZ1_03D_R01.pof")

header=x2.readline()
doc=x2.read()


allDoc=str()
allDoc2=str()

for i in range(1,101):
    nameFile=""
    if(i<10):
        nameFile=cadena+"0"
    else:
        nameFile=cadena

    nameFileX =nameFile+ str(i)+type1
    nameFileY =nameFile+ str(i)+type2

    x2=open(nameFileY)
    header=x2.readline()
    doc=x2.read()
    allDoc=allDoc+doc
    
    y1=open(nameFileX)
    header2=y1.readline()
    doc2=y1.read()
    allDoc2=allDoc2+doc2

print(allDoc2)

fileEND= open("fileEnd_Y.pof",'w+')
fileEND.write(allDoc)
fileEND.close

fileEND_X= open("fileEnd_X.pos",'w+')
fileEND_X.write(allDoc2)
fileEND_X.close