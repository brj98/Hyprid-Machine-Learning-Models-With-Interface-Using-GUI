import numpy as np
import pandas


datalar = np.load("./dataset/diabets.npy", allow_pickle=True)
attr_sayisi=datalar.shape[1]
print(attr_sayisi)
y=datalar[:,attr_sayisi-1]
X=datalar[:,0:attr_sayisi-1]
sifirlar=0
birler=0
i=0
for i in y:
    if i==0:
        sifirlar +=1            
    elif i == 1 :
        birler += 1

print(sifirlar)
print(birler)