import pandas as pd
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import itertools
import pickle


anaKlasor="."


def func_hastaBilgi(y):
    hs,hos=0,0
    for item in y:
        if item==0.0:
            hos+=1
        else:
            hs+=1
    return hs,hos

def func_performansHesapla(cm,scr ,mns):
    TN,TP,FP,FN=cm[0,0],cm[1,1],cm[0,1],cm[1,0]
   
    print ("TP:",TP)
    print ("TN:",TN)#
    print ("FP:",FP)
    print ("FN:",FN)
   
    accuracy=(TP+TN)/(TP+FP+TN+FN)*100
    sensitivity=TP/(TP+FN)*100
    specificity=TN/(TN+FP)*100
    print ("Accuracy:",accuracy," Sen:",sensitivity," Spe",specificity)
    return "Accuracy:",accuracy," Sen:",sensitivity," Spe",specificity,scr ,mns
   
   
def func_verisetiOku(dosyaAdi):

    dataset = pandas.read_csv(anaKlasor+"/dataset/"+dosyaAdi)
    print ("Verisetei bilgisi:",dataset.shape)
   
    datalar=dataset.values
    sutunlar=dataset.columns

    attr_sayisi=dataset.shape[1]
    print ("Öznitelik sayisi:",attr_sayisi)

    y=datalar[:,attr_sayisi-1]
    X=datalar[:,0:attr_sayisi-1]
    sifirlar=0
    birler=0
    i=0
    for index,i in enumerate(y):
        if i==0:
            sifirlar +=1
            if sifirlar > 684 :
                #   datalar.pop(i)
                datalar = np.delete(datalar, index)
                
        elif i == 1 :
            birler += 1
        
    return X,y
def func_model_ciz(model_bilgisi):
    # plotting the metrics
    
    plt.imshow(model_bilgisi, cmap='Blues',interpolation='nearest')
    plt.xticks([0, 1], ['True', 'False'])
    plt.yticks([0, 1], ['True', 'False'])
    for i, j in itertools.product(range(model_bilgisi.shape[0]), range(model_bilgisi.shape[1])):
        plt.text(j, i, model_bilgisi[i, j], horizontalalignment='center', color='black')
    plt.colorbar()
    #plt.show()
    plt.savefig(anaKlasor + "/" + "_confusion_matrix.png")

    print(model_bilgisi, " kayıt bitti...")

def main (index ,ts,rs):
    dosyaAdi="diabetes1.csv"
    X,y=func_verisetiOku(dosyaAdi)
    X1, X_test, y1, Y_test = train_test_split(X, y, test_size=(ts/100), random_state=rs)
    if index==1:
        siniflandirici=DecisionTreeClassifier()
        
    elif index==0:
        siniflandirici=RandomForestClassifier(n_estimators=200)
        
    elif index==2:
        siniflandirici= KNeighborsClassifier(n_neighbors=3)
        
    kf = KFold(n_splits=5)
    scores = cross_val_score(siniflandirici, X, y, cv=kf)
    print("Cross-validation scores:", scores)
    scr="Cross-validation scores:", str(scores)
    print("Mean score:", np.mean(scores))
    mns="Mean score:", str(np.mean(scores))
    
    # Train the model on the full training set
    siniflandirici.fit(X1, y1)
    
    # Make predictions on the test set
    tahminler = siniflandirici.predict(X_test)
    
    # Calculate performance metrics
    cm = confusion_matrix(Y_test, tahminler)
    print(cm)
    basari = accuracy_score(Y_test, tahminler) * 100
    print("Accuracy:", basari)
    func_model_ciz(cm)
    with open("model.pkl", "wb") as file:
        pickle.dump(siniflandirici, file)
    #func_model_ciz("asd")
    #X_train,X_validation,Y_train,Y_validation = train_test_split(X1,y1, test_size=0.8 ,random_state=2)
    #siniflandirici=RandomForestClassifier(n_estimators=200)
    
    # print ("----------------------------------------")
    # print(cross_val_score(siniflandirici ,X1,y1,cv=4))
    
    # tahminler=siniflandirici.predict(X_test)
    # print ("Gerçekler:")
    # print (Y_test)
    
    # print ("Tahminler")
    # print (tahminler)
    
    # cm=confusion_matrix(Y_test,tahminler)
    # print (cm)
    # basari=accuracy_score(Y_test,tahminler)*100
    # func_performansHesapla(cm)
    # print ("Modelin basari:",basari)
    return func_performansHesapla(cm,scr ,mns)

#print(main(0,25,2))
