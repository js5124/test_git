# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:11:42 2021

@author: sujie
"""

#%%

git pull origin master

git add .
git commit -m "change"

git push



#%%virtualenv env
anaconda/env: virtualenv flask1
conda activate flask1
where python
pip list
pip freeze --local
type requirements.txt
conda deactivate
rm -rf 

# del /f /s /q mydir 1>nul
# rmdir /s /q mydir

virtualenv -p /Scripts/python2.6 flask1

python --version 
pip install -r requirements.txt

mkdir dir1
rmdir dir1
rmdir /s dir1 #contents in dir1
cd 
cd ..
cd ../../..
cls
cd "C:\Program Files"

path
tree
exit()

color 0B
color 07

attrib /?
attrib +h file1
attrib r+ -h file1

del file1
echo lalala >test.txt
type test.txt  #show the contents of the file1
echo newline >> test.txt
dir > structure.txt

copy file1 folder1  #file1 in current working folder
del file1

copy file1 C: #copy to the root directory C:
xcopy folder1 folder2  #copy files (not including subdirs) contents of folder1 to folder2
xcopy folder1 folder2 /s #/s means includign subdirs
move folder1 folder2 #move the whole fodler
rename folder2 folder3

dir *.png
file1.png #open a file
ipconfig

/?  #help/option menu


#%%
#corey schafer
import re
print ('\tTab')
print (r'\tTab')
re.compile(r'.')
re.compile(r'\.')
r'Jessie\.com'
r'\s'

r'\bHa'  #Ha HaHa 
r'\BHa'
r'^Start' #Start a string with non-a end
r'end$'
#dash/hypen, period, semicolon, forward slash
r'[0-9]{3}'
r'\d\d\d.\d\d\d.\d\d\d\d' #. include all the words
r'[89]00[._]\d\d\d[._]\d\d\d\d' #character set, just match only one of them
r'[1-5a-zA-Z]'
r'[^a-z]' #^ in character set -- not those after; out of set--- start with; . in set ===. ; . out of set ====any character 
r'[^b]at'  #match mat, not bat
r'\d{3}.\d{3}.\d{4}'
# r'Mr[.]? [A-Z][a-z]+' --wrong
r'Mr\.?\s[A-Z]\w*'#without [A-Z] after, we only match the part before 

#group match
r'M(r|s|rs)\.?\s[A-Z]\w*'
r'(Ms|Mr|Mrs)\.?\s[A-Z]\w*'

r'[a-zA-Z.-]+@[a-zA-Z-]+\.[a-z]{3}'  #match email
r'[a-zA-Z0-9.-]+@[a-zA-Z-]+\.(com|edu|net)'
# r'http(s)?' # do not use group easily
# r'https?://(www.)  #character set . means ., otherwise it means any word; to mean dot, we use \.
r'https?://(www\.)?[a-zA-Z]+'
r'https?://(www\.)?(\w+)(\.\w+)' #just add groups do not change results, group 1-3, group 0 includes all
pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
matches = pattern.finditer(urls)
for match in matches:
    print (match.group(2)) #domain name

subbed = pattern.sub(r'\2\3',urls) #find a match in urls and replace with \2\3, use it in the re.group
pattern.findall(urls) # print a list of strs if no group; print a list of tuples if there are more than 1 group; print group 1 if only 1

# at most one
pattern.match(urls) # can only match those with pattern as the beginning of the string 
pattern.search(urls) # match entire string, print out the first object that match, return None if nothing 

#different flags
re.compile(r'start',re.IGNORECASE) #equal to re.I, other flags like multi-linea



.   --Any character except a new line
\d  -- Digit(0-9)
\D  -- Not a Digit
\w  -- Word (a-z,A-Z,0-9,_)
\W  -- Not a word
\s  -- space, tab, newline
\S  -- Not \s

\b  -- word boundary
\B  


pattern = re.compile(r'abc')
matches = pattern.finditer(text)
for match in matches:
    print (match) #<_sre.SRE_Match object; span=(1,4);match='abc'>


from datetime import datetime
re.match(r'[a-zA-Z0-9_-]+@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}$',s)
s="小明14岁,18上大学"
x=re.search(r'\d+',s).group()
pattern=re.compile('fx_cva_(.*q\d).txt')
pattern1=re.complie(r'[\u4e00-\u9fa5]+')
pattern.match(s)
pattern0=re.match('1\d{9}[0-3,5-7,9]',tel)
pattern0.group()
pattern0.findall()
pattern2=re.match('[\w]{4,20}@163\.com$',email)
if pattern2:
    re=pattern2.group()
pattern3=re.spllit(r':| ',text)
print ("%s log data source %s"%(datetime.now(),"MRKDB"))
result=re.findall(r'daterange=(.*?)%sdkjjf(.*?)&**&',text)  #(.*)
re.sub(r'\d+','100','zhangming: 98')
re.findalll(r'\d+|[a-zA-Z]+',text)
re.findall(r'<div class=(.*)>(.*?)</div>',text)
bool(re.match(r'^(\d{4}|\d{6})$',pin))

#%%
import os
os.chdir(r'C:\\Users\\sujie\\Documents\\Desktop\\scripts')

#decorators
from functools import wraps
import logging
import time
import datetime
# for handler in logging.root.handlers[:]:
#     print (handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# print(repr(logger.handlers))
logger.handlers = [] ## add the line if we cannot find the log file, very important.

file_handler = logging.FileHandler('test2.log')
logger.addHandler(file_handler)
streamhandler = logging.StreamHandler()
logger.addHandler(streamhandler)
# logging.basicConfig(filename ='test.log',level=logging.info)


def my_logger(funcs):
    # print (os.path.join(os.getcwd(),funcs.__name__+'.log'))
    # logging.FileHandler(os.path.join(os.getcwd(),funcs.__name__+'.log'))
    # logging.basicConfig(filename =os.path.join(os.getcwd(),funcs.__name__+'.log'),level=logging.info)
    # @wraps(funcs)
    def wrapper(*args,**kwargs):
        print ("getting into logger wrapper ")
        logger.info(f'{args[0]} log as attached: {funcs.__name__} func is used')  #logging.info != logging.INFO  # args[0]
        print ("wrapper executed")
        # return funcs(*args,**kwargs)
        funcs(*args,**kwargs)
    print ("get into logger func")
    return wrapper

def timer(funcs):
    # @wraps(funcs)
    def wrapper(*args,**kwargs):
        print ("start counting time")
        logger.info(f'{args[0]} log for timer as attached: {funcs.__name__} func is used')  #logging.info != logging.INFO  # args[0]
        start = time.time()
        funcs(*args,**kwargs)
        print ("timer executed")
        # print (datetime.datetime.fromtimestamp(time.time()-start))
        print (time.time()-start)
    return wrapper


@timer
@my_logger   #both stop before running the func, 
def funcs(indx):   #only execute once, first logging, then decorator inside, then wrapper, then another wrapper, then function itself
    print ("get into func itself")
    time.sleep(1)
    print ('{} started'.format(indx))

if __name__ == '__main__':
    funcs('test')


#%%
import os
os.getcwd()
os.chdir(r'C:\\Users\\sujie\\Documents\\Desktop\\test')

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#prit plt.style.available
# plt.style.use('seaborn')
# plt.style.use('fivethirtyeight')
plt.style.use('ggplot')

plt.xkcd() #graphic comic

data = pd.read_csv('data.csv')
ages = data['Age']
dev_salaries = data['All_Devs']
py_salaries = data['Python']
js_salaries = data['JavaScript']

###subplots
#fig,(ax1,ax2) = plt.subplots(nrows=2.ncols=1,sharex = True)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax1.plot(ages, dev_salaries, color='#444444', #red,green,blue  
         linestyle='--', label='All Devs')

ax2.plot(ages, py_salaries,'k--',label='Python')
ax2.plot(ages, js_salaries,color = 'b',linestyle='-', marker='o',linewidth = 3, label='JavaScript')  #format:marker color line, linewidth default 1
 
ax1.legend()
ax1.set_title('Median Salary (USD) by Age')
ax1.set_ylabel('Median Salary (USD)')

ax2.legend()
ax2.set_xlabel('Ages')
ax2.set_ylabel('Median Salary (USD)')

###bar chart
x_indexes = np.arange(len(ages))
width = 0.25 #default 0.8
plt.bar(x_indexes, py_salaries,width = width)
plt.bar(x_indexes+width,js_salaries,width = width)  # if using the same x_indexes, the bars will be stacked together
plt.xticks(ticks = x_indexes, labels=ages)
plt.barh()  #show a lot of bar charts, like sorting them. easier to show it in barh

###pie chart
slices = [60,40]
labels = ['sixty','forty']
colors = ['#008fd5','#fc4f30']
explode = [0,0.1]
plt.pie(slices,labels = labels,wedgeprops={'edgecolor':'black'},explode = explode,shadow=True,startangle=90,autopct ='%1.1f%%')

#stackplot 
plt.stackplot(minutes,player1,player2,player3,labels=['1','2','3'])
plt.legend(loc='upper left') #loc=(0.07,0.05) #7% to the left

#filling area on line plots
plt.fill_between(ages,py_salaries,overall_median,where = (py_salaries>js_salaries),interpolate = True, color = 'red', alpha=0.2,legend ='above') 
plt.fill_between(ages,py_salaries,overall_median,where = (py_salaries<js_salaries),alpha=0.2) 

## histograms
ages = [18,23,32,56]
bins = [10,20,30,40,50,60]
plt.hist(ages,bins=bins,edgecolor='black',log=True)
plt.axvline(29,color='red',label='age median',linewidth=2)

##scatter plot
plt.scatter(x,y,s=100,marker='o',c='green')
plt.scatter(x,y,s=sizes,marker='o',cmap='Greens')
plt.xscale('log')
plt.yscale('log')
cbar = plt.colorbar()
cbar.set_label('Satisfaction')


##plot time series
from matplotlib import dates as mpl_dates
pd.plot_date(dates,y,linestyle='solid')
plt.gcf.autofmt_xdate() #get current figure

date_format - mpl_dates.DateFormatter('%b,%d %Y')
plt.gca.xaxis.set_major_formatter(date_format)

##sns
sns.countplot(x='Survived',hue='Pclass',data=df)
plt.show()

plt.grid(True)
plt.tight_layout()
plt.show()

fig1.savefig('fig1.png')
fig2.savefig('fig2.png')


#%%
import csv
csv_reader = csv.DictReader(csv_file) #use it as keys instead of index
row = next(csv_reader)  # OrderedDict, semicolon

with open('data.txt','r',encoding='utf-8') as f:
    data = f.read()


#%%
from collections import Counter
c = Counter(['Python','C++'])
c.update(['Python','Java'])
c.most_common(15) #item{key:value}
alist.reverse() #inplace automatically


#%% Error
def divide(x,y):
    if y==0:
        raise ValueError ('cannot !')
    return x/y


#%%
import unittest
import calc #module defining add function
import Dog
import requests
from unittest.mock import patch

class TestCalc(unittest.TestCase):
    
    def test_add(self):  #test should be used as the start; do not run in order for these functions
        result = calc.add(10,5)
        self.assertEqual(result,14)
        
        # self.assertEqual(calc.add(10,5),14)
        self.assertEqual(calc.add(-1,1),0)  #three tests --- one dot in the console result, dot decided by the test function number
        
    def test_divide(self):
        self.assertEqual(calc.subtract(1,-1),2)
        
        self.assertRaises(ValueError,calc.divide,10,0)  #will pass
        self.assertRaises(ValueError,calc.divide,10,2)  #will break
    
        with self.assertRaises(ValueError):  #context manager
            calc.divide(10,0)
        
        

class TestDog(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):  #run at the beginning
        print ('setupClasss')
    
    @classmethod 
    def tearDownClass(cls):  #run it at the end of the test
        print ('teardownClass')
        
    
    def setUp(self):  #U instead of u, D instead of d
        # pass
        self.dog1 = Dog('Pomeranian')
        self.dog2 = Dog('KC')
    
    def tearDown(self):
        # pass
        print ("teardown\n")
    
    
    def test_update_name(self):  #test should be used as the start
        self.dog1.update_name()
        self.dog2.update_name()
        
        self.assertEqual(self.dog1.name,'Pomeraniann')
        self.assertEqual(self.dog2.name,'KCC')
    
    def test_monthly_schedule(self):
        with patch('Dog.requests.get') as mocked_get:
            mocked_get.return_value.ok = True
            mocked_get.return_value.text = 'Success'
            
            schedule = self.dog2.monthly_schedule('May')
            mocked_get.assert_called_with('http://company.com/KC/May')
            self.assertEqual(schedule,'Success')

            mocked_get.return_value.ok = False
            
            schedule = self.dog2.monthly_schedule('June')
            mocked_get.assert_called_with('http://company.com/KC/June')
            self.assertEqual(schedule,'Bad Request')


if __name__ = '__main__':  #python test_calc.py directly is fine//running in editor is fine//otherwise in console # python -m unittest test_calc.py
    unittest.main()
    

#%%
# class 
# inheritance
class Veichle():
    
    number_of_v = 0
    
    def __init__(self,color, price):
        self.color = color
        self.price = price

class car(Veichle):
    def __init__(self,color,price,speed):
        super().__init__(color,price)
        self.speed = speed
    def beep(self):
        print ("Honk")


#overriding
class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __add__(self,p):   #__sub__. __mul__
        return Point(self.x+p.x,self.y+p.y)
    def __str__(self):   # when printing, goes to check if it has this attr
        return "("+str(self.x)+","+str(self.y)+")"
    def length(self):
        import math
        return math.sqrt(self.x**2+self.y**2)
    def __gt__(self,p):
        return self.length()>p.length()
    def __ge__(self,p):
        return self.length()>=p.length()
    def __lt__(self,p):
        return self.length()<p.length()
    def __le__(self,p):
        return self.length()<=p.length()
    def __ee__(self,p):
        return self.x==p.x and self.y==p.y
    
    
    
Point(1,2)+Point(3,4)
print (Point(1,2)+Point(3,4))

p1 = Point(1,2)
p2 = Point(3,4)

print (p1==p2)
print (p1<p2)

#%%
#static class
class Dog:
    dogs = []   # class variable
    def __init__(self,name):
        self.name = name
        self.dogs.append(self)
    
    @classmethod  # decorator
    def num_dogs(cls):
        return len(cls.dogs)
    
    @staticmethod
    def bark(n):   # seperate from self, methods, only irrelevant variable n
        for _ in range(n):
            print ("bark")
    
    @property
    def nick_name(self):
        return 'little '+self.name
    
    def update_name(self):
        self.name+=self.name[-1]
        
    def monthly_schedules(self.month):
        responss = requests.get(f'http://company.com/{self.name}/{month}')
        if response.ok:
            return response.text
        else:
            return 'Bad Request'
        
tim = Dog('Tim')
tim.num_dogs()

# do not need to create an instance
Dog.bark(5)   
Dog.num_dogs()


#%%

# private and public class
class _Private():    #_ means private
    def __init__(self,name):
        self.name = name


class NonPrivate():
    def __init__(self,name):
        self.name = name
        self.priv = _Private(name)
    
    def _display(self):
        print ("Do not use it outside the funtion")
    
    def display(self):
        print ("Hi")

#%% pandas
import pandas as pd
import numpy as np
#pandas

df = pd.DataFrame()
df.iloc[1:3]
df[1:3]   # row name as 1,2,3,4
df[col1].value_counts()
df.drop(index = 1)
df.drop(columns = 'col1')
pd.merge(df1,df2,how='outer',left_index = True,right_index = True)
df1.join(df2,lsuffix ='_x',rsuffix = '_y')
pd.concat([df1,df2],axis=1)
pd.concat(df.groupby())
df1.append(df2)  # merge vertically 
df1.append(dict2,ignore_index=True) #dict2 keys = df1.columns, == append one row in dataframe
df1.loc[i]=[a,b,c]


df[['First','Last']] = df['full_name'].str.split(' ',expand = True)
df.append({'first':'Tony'},ignore_index=True)
df = df.append(df2,ignore_index=True, sort=False)  #不同列数目
df.drop(index=4)
df.drop(index = df[df['Last']=='Goldman'].index)
df.drop(columns = [df.columns[0:3]])
df['first'].value_counts(normalize=True)
pd.set_option('display.max_columns',85)
df.sort_index(ascending=True)
d_parser = lambda x: pd.datetime.strptime(x,'%Y-%m-%d %I-%p')
pd.read_csv('.csv',index_col=col0,na_values=['NA','Missing'],parse_dates=['Date'],date_parser = d_parser)
df.loc[~filt,'email']
df['first'].isin(['john','david'])
df['first'].str.contains('Python',na)

grps = df.groupby(['Country'])
grps.get_group('India')

df.sort_values(by=col1,ascending=True)
df.dropna(axis='index',how='any') #default
df.dropna(axis='index',how='any',subset=['first','email'])
df.replace('NA',np.nan,inplace = True)
df.fillna(0)

df['age'] =df['age'].astype(float) #not int, #missing value/nontype cannot be coverted to integer, do not use fillna with 0 and then take average may be affected.
df['Date'] = pd.to_datetime(df['Date'],format = '%Y-%m-%d %I-%p')
date0.day_name()  #Friday
df['DayOfWeek'] = df['Date'].dt.day_name()
df[df['Date']>='2020']
df[df['Date']>=pd.to_datetime('2019-01-01')]
df['2019']
df['2020-01':'2020-02'] #2020-01-01---------2020-02-28
df['High'].resample('D').max()
df.resample('W').agg({'High':max,'Volume':sum})
# %matplotlib inline
# pip install openpyxl xlrd
df.to_json('data.json',orient='records',lines=True)  #per line, a dict consisting all the columns as keys, records mean list-like
pd.read_json('data.json',orient='records',lines=True)
# pip install SQLAlchemy #object relational mapper
# pip install psycopg2-binary # posgretal
from sqlalchemy import create_engine
import psycopg2
engine = create_engine('postgresql://dbuser:dbpass@localhost:5432/sample_db')
india_df.to_sql('sample table',engine)  #create a table
india_df.to_sql('sample table',engine,if_exists='replace')  #create a table
pd.read_sql('sample table',engine,index_col=col1)
pd.read_sql_query('SELECT * FROM sample_table where ..',engine,index_col=col1)

df1 = pd.read_json('https://www.youtube.com/watch?v=N6hyN6BW6ao&list=RDCMUCCezIgC97PvUuR4_gbFUs5g&index=4') #cannot be any http, must contain strings only
df.apply(len,axis='columns')
df.applymap(len)