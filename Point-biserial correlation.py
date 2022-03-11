#coding=utf-8

from array import array
from scipy.stats import pearsonr
import csv
import numpy as np
import scipy.stats as stats

basedir = '/Users/guanmiaomiao/Documents/1_Project/5_ToxCast_Model/4_数据整理/'

with open(basedir + 'Point-biserial correlation_data.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        column_EDC_LD50 = [row[0] for row in reader]

for i in range(143):
        with open(basedir + 'Point-biserial correlation_data.csv','r') as csvfile:
                reader = csv.reader(csvfile)
                column_EDC_chit = [row[i+1] for row in reader]
        #print(np.array(column_EDC_chit))
        x = list(map(int,column_EDC_LD50))
        y = list(map(int,column_EDC_chit))
        r, p=stats.pearsonr(x,y)
        print(r,p)
        fp = open(basedir + 'Point-biserial correlation_result.csv', 'a')
        fp.write(str(i+1)+',')
        fp.write(str(r)+',')
        fp.write(str(p)+'\n')
        fp.close()
 
# 输出:(r, p)
# r:相关系数[-1，1]之间
# p:p值越小

