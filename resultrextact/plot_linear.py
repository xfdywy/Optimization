import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import shutil

os.chdir(r'C:/Users/v-yuewng/Downloads/lrgdsgd/code/resultextract/linear regression/20160901/')

os.getcwd()


#shutil.copy('D:/project/linear_l2_gd/linear_l2_gd/x64/Release/linear_simu_gd.csv',r'C:/Users/v-yuewng/Downloads/lrgdsgd/code/resultextract/linear regression/20160901/linear_gd.csv')
#shutil.copy('D:/project/linear_l2_sgd/linear_l2_sgd/x64/Release/linear__simu_sgd.csv',r'C:/Users/v-yuewng/Downloads/lrgdsgd/code/resultextract/linear regression/20160901/linear_sgd.csv')
#shutil.copy('D:/project/linear_l2_svrg/linear_l2_svrg/x64/Release/linear_simu_svrg.csv',r'C:/Users/v-yuewng/Downloads/lrgdsgd/code/resultextract/linear regression/20160901/linear_svrg.csv')


sgdloss = pd.read_csv('linear_sgd.csv')
gdloss = pd.read_csv('linear_simu_gd.csv')
svrgloss = pd.read_csv('linear_svrg.csv')





thr = max(max(sgdloss.passeddata),max(gdloss.passeddata),max(svrgloss.passeddata))

sgdloss_f = sgdloss[sgdloss.passeddata<thr]
gdloss_f = gdloss[gdloss.passeddata<thr]
svrgloss_f = svrgloss[svrgloss.passeddata<thr]

sgdloss_f = sgdloss[sgdloss.passeddata%10000==0]
gdloss_f = gdloss[gdloss.passeddata%10000==0]
svrgloss_f = svrgloss[svrgloss.passeddata%10000==0]

trainopt  = min(gdloss.trainloss.min(),svrgloss.trainloss.min(),sgdloss.trainloss.min())
testopt = gdloss.testloss.min()

trainopt  =0.99104323
testopt = 0.9859887
def plot_passeddata_q(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f):
    plt.hold(True)
    plt.plot(data1.passeddata,data1.trainloss,'b')
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
    plt.plot((data3.passeddata*1.1),np.log10(data3.trainloss-trainopt),'g-')
    plt.hold(False)
    plt.xlim(0,7000000)
    plt.ylim(-8,2)
    plt.legend(['sgd','gd','svrg'],loc = 4, fontsize=21)

    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.title("simulation data", fontsize=25)
    plt.xlabel("passed data ( 1e4 )", fontsize=21)
    plt.ylabel("train loss - optimum(10^)", fontsize=21)
    ax=plt.gca()     
    ax.set_xticks(range(0,7500000,1000000)) 
    ax.set_xticklabels(range(0,750,100))  
    
    plt.savefig("traiinloss",bbox_inches='tight',dpi=200)
    
    
    
def plot_passeddata_test(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f,testopt=testopt):
   
    plt.hold(True)
    plt.style.use('bmh')
    plt.plot((data1.passeddata),np.log10(data1.testloss),'b-o')
    plt.plot((data2.passeddata),np.log10(data2.testloss),'r-s')
    plt.plot((data3.passeddata),np.log10(data3.testloss),'g<-')
    plt.xlim(0,3000000)
    plt.ylim(-0.5,2)
    
    plt.legend( ['sgd','gd','svrg'],loc =1,fontsize=21)
    ax=plt.gca()     
    ax.set_xticks(range(0,3500000,500000)) 
    ax.set_xticklabels(range(0,350,50))  
    ax.set_yticks( np.array(range(-1,4))/2.0) 
    ax.set_yticklabels( np.array(range(-1,4))/2.0)  
    plt.xticks(fontsize=21) 
    plt.yticks(fontsize=21) 
    plt.title("simulation data", fontsize=25, )
    plt.xlabel("passed data(1e4) ", fontsize=21)
    plt.ylabel("test loss (10^)", fontsize=21)

    plt.hold(False)
    plt.savefig("testloss")
    
    
def plot_passeddata_test_nolog(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f,testopt=testopt):
    plt.hold(True)
    plt.style.use('bmh')
    plt.plot((data1.passeddata),(data1.testloss),'b--')
    plt.plot((data2.passeddata),(data2.testloss),'r-.')
    plt.plot((data3.passeddata),(data3.testloss),'g-')
    plt.xlim(0,2000000)
    plt.ylim(0,50)
    
    plt.legend( ['sgd','gd','svrg'],loc =1, fontsize=21)
    
    plt.title("simulation data", fontsize=25)
    plt.xlabel("passed data ( 1e4 ) ", fontsize=21)
    plt.ylabel("test loss", fontsize=21)
    ax=plt.gca()     
    ax.set_xticks(range(0,2500000,500000)) 
    ax.set_xticklabels(range(0,350,50))      
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.hold(False)
    plt.savefig("testloss_nolog",bbox_inches='tight',dpi=200)
    
def plot_passeddata_train_nolog(data1 =sgdloss_f ,data2=gdloss_f,data3=svrgloss_f,trainopt=trainopt):
    plt.style.use('bmh')
    plt.hold()
    plt.plot((data1.passeddata),(data1.trainloss),'b--')
    plt.hold('on')
    plt.plot((data2.passeddata),(data2.trainloss),'r-.')
    plt.plot((data3.passeddata),(data3.trainloss),'g-')
    plt.hold(False)
    plt.xlim(0,1000000)
    plt.ylim(0,50)
    plt.legend(['sgd','gd','svrg'],loc = 0, fontsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.title("simulation data", fontsize=25)
    plt.xlabel("passed data ( 1e4 )", fontsize=21)
    plt.ylabel("train loss ", fontsize=21)
    ax=plt.gca()     
    ax.set_xticks(range(0,1500000,500000)) 
    ax.set_xticklabels(range(0,350,50))      
    
    plt.savefig("traiinloss_nolog",bbox_inches='tight',dpi=200)

#plot_passeddata_q()
#plot_numiter()
#plot_passeddata_train()
plot_passeddata_train_nolog()

#plot_passeddata_test()
#plot_passeddata_test_nolog()




