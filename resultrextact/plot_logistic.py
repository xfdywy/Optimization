import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import shutil

os.chdir(r'C:/Users/v-yuewng/Downloads/lrgdsgd/code/resultextract/logistic regression/20160901/')

os.getcwd()


#shutil.copy('C:/Users/v-yuewng/Downloads/lrgdsgd/code/logistic_l2_gd/logistic_l2_gd/x64/Release/logistic_rcv1t_gd.csv',r'C:/Users/v-yuewng/Downloads/lrgdsgd/code/resultextract/logistic regression/20160901/logistic_gd.csv')
#shutil.copy('C:/Users/v-yuewng/Downloads/lrgdsgd/code/logistic_l2_sgd/x64/Release/logistic_rcv1_sgd.csv',r'C:/Users/v-yuewng/Downloads/lrgdsgd/code/resultextract/logistic regression/20160901/logistic_sgd.csv')
shutil.copy('D:/project/mysvrg/mysvrg/x64/Release/logistic_rcv1_svrg.csv',r'C:/Users/v-yuewng/Downloads/lrgdsgd/code/resultextract/logistic regression/20160901/logistic_svrg.csv')


sgdloss = pd.read_csv('logistic_sgd.csv')
gdloss = pd.read_csv('logistic_gd.csv')
svrgloss = pd.read_csv('logistic_svrg.csv')










sgdloss_f = sgdloss[sgdloss.passeddata%10000==0]
gdloss_f = gdloss[gdloss.passeddata%20242==0]
svrgloss_f = svrgloss[(svrgloss.passeddata)%10000<800]
svrgloss_f.loc[1,'passeddata']=0


trainopt  = min(gdloss.trainloss.min(),svrgloss.trainloss.min(),sgdloss.trainloss.min())+1e-8
testopt =  min(gdloss.testloss.min(),sgdloss.testloss.min(),svrgloss.testloss.min())
print(trainopt)
def plot_passeddata_q(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f):
    plt.hold(True)
    plt.plot(data1.passeddata,data1.trainloss)
    plt.plot(data2.num_iter,data2.trainloss,'r')
    plt.plot(data3.passeddata,data3.trainloss,'g')
    plt.xlim(0,10000)
    plt.ylim(0,0.7)
    plt.legend(['sgd','gd','svrg'])
    plt.hold(False)
    
    

def plot_passeddata_train(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f,trainopt=trainopt):
    plt.style.use('bmh')
    plt.hold()
    plt.plot((data1.passeddata),np.log10(data1.trainloss-trainopt),'b--')
    plt.hold('on')
    plt.plot((data2.passeddata),np.log10(data2.trainloss-trainopt),'r-.')
    plt.plot((data3.passeddata),np.log10(data3.trainloss-trainopt),'g-')
    plt.hold(False)
    plt.xlim(0,1500000)
    plt.ylim(-6,1)
    plt.legend(['sgd','gd','svrg'],loc = 1, fontsize=21)
   
    plt.title("rcv1 data", fontsize=25)
    plt.xlabel("passed data (1e4) ", fontsize=21)
    plt.ylabel("train loss - optimum(10^)", fontsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize = 21)
    ax=plt.gca()     
    ax.set_xticks(range(0,1600000,200000)) 
    ax.set_xticklabels(range(0,160,20))      
    ax.set_yticks(range(-6,1)) 
    ax.set_yticklabels(range(-6,1)) 
    plt.savefig("trainloss",bbox_inches='tight',dpi=200)
    
    
    
def plot_passeddata_test(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f,testopt=testopt):
    plt.hold(True)
    plt.style.use('bmh')
    plt.plot((data1.passeddata),np.log10(data1.testloss-testopt),'b-')
    plt.plot((data2.passeddata),np.log10(data2.testloss-testopt),'r-')
    plt.plot((data3.passeddata),np.log10(data3.testloss-testopt),'g-')
    plt.xlim(0,1500000)
    plt.ylim(-7,2)
    
    plt.legend( ['sgd','gd','svrg'],loc =3)
    
    plt.title("logistic regression result (rcv1 data)", fontsize=25)
    plt.xlabel("passed data( 1e4 )", fontsize=21)
    plt.ylabel("test loss - optimum(10^)", fontsize=21)
    plt.hold(False)
    plt.savefig("testloss",bbox_inches='tight',dpi=200)

def plot_passeddata_test_nonlog(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f,testopt=testopt):
    plt.hold(True)
    plt.style.use('bmh')
    plt.plot((data1.passeddata),(data1.testloss),'b--')
    plt.plot((data2.passeddata),(data2.testloss),'r-.')
    plt.plot((data3.passeddata),(data3.testloss),'g-')
    plt.xlim(0,500000)
    plt.ylim(0,0.8)
    plt.xticks( fontsize=21)
    plt.yticks( fontsize=21)
    plt.legend( ['sgd','gd','svrg'],loc =1, fontsize=21)
    ax=plt.gca()     
    ax.set_xticks(range(0,600000,100000)) 
    ax.set_xticklabels(range(0,60,10))  
    ax.set_yticks(np.array(range(1,10,2))/10.0) 
    ax.set_yticklabels(np.array(range(1,10,2))/10.0)      
    plt.title("rcv1 data", fontsize=25)
    plt.xlabel("passed data ( 1e4 )", fontsize=21)
    plt.ylabel("test loss ", fontsize=21)
    plt.hold(False)
    plt.savefig("testloss",bbox_inches='tight',dpi=200)

   
    

def plot_passeddata_train_nolog(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f,trainopt=trainopt):
   
    plt.hold()
    plt.plot((data1.passeddata),(data1.trainloss),'b--')
    plt.hold('on')
    plt.plot((data2.passeddata),(data2.trainloss),'r-.')
    plt.plot((data3.passeddata),(data3.trainloss),'g-')
    plt.hold(False)
    plt.xlim(0,500000)
    plt.ylim(0,0.8)
    plt.xticks( fontsize=21)
    plt.yticks( fontsize=21)
    plt.legend(['sgd','gd','svrg'],loc = 0, fontsize=21)
    ax=plt.gca()     
    ax.set_xticks(range(0,600000,100000)) 
    ax.set_xticklabels(range(0,60,10))      
    ax.set_yticks(np.array(range(1,10,2))/10.0) 
    ax.set_yticklabels(np.array(range(1,10,2))/10.0)  
    plt.title("rcv1 data", fontsize=25)
    plt.xlabel("passed data ( 1e4 )", fontsize=21)
    plt.ylabel("train loss ", fontsize=21)
    
    plt.savefig("traiinloss_nolog",bbox_inches='tight',dpi=200)

#plot_passeddata_q()
#plot_numiter()
#plot_passeddata_train()
plot_passeddata_train_nolog()
#plt.hold(False)
#plot_passeddata_test_nonlog()
#plt.hold(False)



