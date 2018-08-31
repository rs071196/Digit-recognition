from sklearn import svm,datasets,metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mimg

data=datasets.load_digits()
train_sec=[50,70,80]; 
section_val=[]
train_label=['50:50','70:30','80:20']
kernle=['poly','rbf','linear']; 
acc=np.zeros((3,3))
plt.figure(1)
for i in range(1,20):
    im=data.data[i]
    im=im.reshape(8,8)
    plt.subplot(2,10,i)
    plt.imshow(im)
    #plt.axes('None')
    
plt.show()

for i in range(0,3):

    
    data_total=data.data.shape[0]
    
    train_section=(train_sec[i]*data_total)//100    
    section_val.append(train_section)
    train_data=data.data[0:train_section,:]
    train_target=data.target[0:train_section]

    test_data=data.data[train_section:,:]
    test_target=data.target[train_section:]
    for j in range(len(kernle)):

        svm_model=svm.SVC(kernel=kernle[j])
        svm_model.fit(train_data,train_target)
        
        output=svm_model.predict(test_data)
        
        acc[i][j]=metrics.accuracy_score(test_target,output)*100
        
dataframe=pd.DataFrame(acc,index=train_label,columns=kernle)
print("dataframe with thier accuracy: ")
print(dataframe)









