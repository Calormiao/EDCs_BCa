#coding=utf-8
#1、训练，生成模型文件
#2、预测

from distutils import filelist
from itertools import count
import os
from pickle import FALSE, TRUE
from site import addsitedir
from ssl import ALERT_DESCRIPTION_ACCESS_DENIED
from tkinter.tix import COLUMN
import numpy
import pandas
import tensorflow
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
#from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from os.path import exists
import time

#设置随机种子，可复现结果
seed = 12061204
numpy.random.seed(seed)

#文件路径
basedir = '/Users/guanmiaomiao/Documents/1_Project/5_ToxCast_Model/3_Model_MLP/'
Input_dir = os.path.join(basedir,'1_input')
Model_dir = os.path.join(basedir,'2_model')
Output_dir = os.path.join(basedir,'3_output')

#判断文件夹内是否存在系统文件
def issystemfile(Input_dir):
    if exists(os.path.join(Input_dir,'.DS_Store')):
        systemfile=os.path.join(Input_dir,'.DS_Store')
        os.remove(systemfile)

#程序运行日志
Log_file = open(basedir + 'Model_log.txt' , 'a')
Log_file.write(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
Log_file.write('程序运行开始:\n')

print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()),'程序运行开始:\n')

#文件路径下需读取文件数目
filelist = os.listdir(Input_dir)
print(filelist)
Log_file.write('filelist为:\n')
for i in range(len(filelist)):
    Log_file.write(str(i) + ' ' + str(filelist[i]) + '\n')
Log_file.close()

#with open:输入文件；fp = open：输出文件     
#建立模型，设置参数
def create_model():
    #序列模型 Sequential，序列模型各层之间是依次顺序的线性关系
    model = Sequential() 
    #逐层添加网络结构
    #输入层
    model.add(Dense(2048, input_dim=2048, kernel_initializer='normal', activation='relu'))
    #隐藏层*2
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(100, activation='relu'))
    #输出层
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #编译模型，指定损失函数，优化器，度量
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Record_Results():
    datadir = '/Users/guanmiaomiao/Documents/1_Project/5_ToxCast_Model/2_ToxCast_Assaydata2input/1_Target2Assay/'
    Assay2Chnm_dir = os.path.join(datadir,'2_output')

    #获取试验名称 
    with open(Assay2Chnm_dir + "/Target2Assay.txt",'r') as csvfile:
        reader = csv.DictReader(csvfile)
        column_assay = [row['Assay'] for row in reader]

    #获取试验id
    with open(Assay2Chnm_dir + "/Target2Assay.txt",'r') as csvfile:
        reader = csv.DictReader(csvfile)
        column_aeid = [row['aeid'] for row in reader]

    for i in range(len(column_assay)):
        if input_name == column_assay[i] + '.csv':
            aeid = column_aeid[i]

    fp = open(Model_dir + "/result.csv",'a')
    #title_list ='aeid,Assay,Total_Chemicals, Active, InActive, Model_accuacy, tpr, fpr'
    #fp.write(title_list + '\n')
    fp.write(aeid + ',')
    fp.write(input_name + ',')
    fp.close()

#5折交叉验证评估模型
def kfold_val(x_train_res, y_train_res,count,j):
    estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=5)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, x_train_res, y_train_res, cv=kfold)
    Model_accuacy = results.mean()*100
    #print("Results: %.2f%% (%.2f%%)" % (Model_accuacy, results.std()*100))

    y_pred = cross_val_predict(estimator, x_train_res, y_train_res, cv=kfold)
    #混淆矩阵
    conf_mat = confusion_matrix(y_train_res, y_pred)
    
    Total_Chemicals = count
    Active = j
    InActive = count - j
    tpr = conf_mat[0][0]/ (conf_mat[0][0] + conf_mat[0][1]) #TPR=TP/(TP+FN)
    fpr = conf_mat[1][0]/ (conf_mat[1][0] + conf_mat[1][1]) #FPR=FP/(FP+TN)

    info_list = [Total_Chemicals, Active, InActive, Model_accuacy, tpr, fpr] #化合物总数，活性，非活性，模型精确度，真正例率，假正例率

    Record_Results()
    fp = open(Model_dir + "/result.csv",'a')
    for i in range(len(info_list)):
        fp.write(str(info_list[i])+',')
    fp.write('\n')
    fp.close()

def main(input_name):
    Log_file = open(basedir + 'Model_log.txt' , 'a')
    Log_file.write(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
    Log_file.write(input_name+ '\n')
    Log_file.close()
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()),input_name)

    count = 0; j =0
    with open(Input_dir + '/' + input_name,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        column_smiles = [row['Smiles'] for row in reader]

    with open(Input_dir + '/' + input_name,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        column_type = [row['Type'] for row in reader]

    mols =[]
    fps = []
    column_type_arrange = []
    n = 0 #有摩根指纹的化合物数目
    for i in range(len(column_type)):
        mol = Chem.MolFromSmiles(column_smiles[i])
        if mol != None:
            column_type_arrange.append(column_type[i])
            mols.append(mol)
            if column_type[i] == 'True': 
                j = j + 1
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mols[n], 2))
            n = n + 1
        count = count + 1
        
    #对类进行编码
    #FALSE(非活性) = 0, TRUE(活性) = 1 
    encoder = LabelEncoder()
    print(encoder)
    encoder.fit(column_type_arrange)
    enc_y = encoder.transform(column_type_arrange)
    print(enc_y)
    
    if j >=6 and count-j >=6:
        for i in range(len(enc_y)):
            print(enc_y[i])
            if enc_y[i] == 1: #分了两类
                fps = numpy.array(fps)
                sm = SMOTE(random_state=12, sampling_strategy = 'minority')
                x_train_res, y_train_res = sm.fit_resample(fps, enc_y) #result data

                #调用模型函数
                model = create_model()
                #训练模型
                model.fit(x_train_res, y_train_res, epochs=5, batch_size=5)
                #保存模型
                model.save(Model_dir + '/' + input_name + '_model.h5')
                #交叉验证评估模型
                kfold_val(x_train_res, y_train_res,count,j)
                break
    else:
        print('某分类化合物数目<6,不适用建立模型')

fp = open(Model_dir + "/result.csv",'a')
title_list ='aeid,Assay,Total_Chemicals, Active, InActive, Model_accuacy, tpr, fpr'
fp.write(title_list + '\n')
fp.close()

if __name__ == "__main__":
    for input_name in filelist :
        main(input_name)

#程序运行日志
Log_file = open(basedir + 'Model_log.txt' , 'a')
Log_file.write(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
Log_file.write('程序运行结束\n')

print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()),'程序运行结束\n')