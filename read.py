import pandas
import numpy as np

df = pandas.read_csv('./dataset/diabetes1.csv')
#print(df)
datalar=df.values
attr_sayisi=datalar.shape[1]
y=datalar[:,attr_sayisi-1]
X=datalar[:,0:attr_sayisi-1]
#print(X)
#print(y)
sifirlar=0
birler=0
i=0

for index,i in enumerate(y):
    if i==0:
        sifirlar +=1
        if sifirlar < 632 :
            #   datalar.pop(i)
            datalar = np.delete(datalar, index,axis=0)
            
    elif i == 1 :
        birler += 1

datalar = np.array(datalar)
np.save("./dataset/diabets.npy", datalar)
print(sifirlar)
print(birler)
#df = df.drop('column_name', axis=1)