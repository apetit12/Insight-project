#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoinepetit
"""

file_name = 'bq-results-20191005-105305-questions.csv'

#------------------------------------------------------------------------------
########################## LIBRARY IMPORT  ####################################
#------------------------------------------------------------------------------
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, SCORERS
#print(sorted(SCORERS.keys()))  # for all scorer options
#from sklearn.linear_model import ElasticNet
from sklearn.tree import export_graphviz
#import statsmodels.api as sm

#------------------------------------------------------------------------------
################################ FUNCTIONS ####################################
#------------------------------------------------------------------------------
def has_ad(metadata):
    '''
    Search metadata to find ad information
    '''
    if 'hasAd' not in metadata:
        return 0
    else:
        idx = metadata.find('hasAd')
        if metadata[idx+7]=='f':  # no ad
            return 0
        elif metadata[idx+7]=='t':  # ad
            return 1
        else:
            raise ValueError("Should be true of false", metadata[idx+7])

def has_multiplier(metadata):
    '''
    Search metadata to find multiplier bonus information
    '''
    if 'pointsMultiplier' not in metadata:
        return 0
    else:
        idx = metadata.find('pointsMultiplier')
        try:
            return int(metadata[idx+18:idx+20])
        except:
            try:
                return int(metadata[idx+18:idx+19])
            except:
                raise ValueError('wrong multiplier number', metadata[idx+16:idx+17])

def is_weekday(num):
    '''
    Classifies days of week into weekday (1) or weekend (0)
    '''
    if num<5:
        return 1
    else:
        return 0

def get_time_of_day(dt):
    '''
    Convert datetime into period of the day: morning/afternoon/evening/night
    '''
    dt_time = datetime.datetime.time(dt.tz_localize('utc').tz_convert('US/Eastern'))
    if dt_time >= datetime.time(5,00,00) and dt_time < datetime.time(12,00,00):
        return 'morning'
    elif dt_time >= datetime.time(12,00,00) and dt_time < datetime.time(17,00,00):
        return 'afternoon'
    elif dt_time >= datetime.time(17,00,00) and dt_time < datetime.time(21,00,00):
        return 'evening'
    elif dt_time >= datetime.time(21,00,00) or dt_time < datetime.time(5,00,00):
        return 'night'
    else:
        raise ValueError("Wrong time %s" %str(dt))

def get_period(dt):
    '''
    Divide time horizon into time periods to account from time-dependent exogeneous factors
    '''
    dt_date = dt.to_pydatetime()
    if dt_date<datetime.datetime.strptime('2018-01-01','%Y-%m-%d'):
        return 1
    elif dt_date>=datetime.datetime.strptime('2018-01-01','%Y-%m-%d') and dt_date<datetime.datetime.strptime('2018-07-01','%Y-%m-%d'):
        return 2
    elif dt_date>=datetime.datetime.strptime('2018-07-01','%Y-%m-%d') and dt_date<datetime.datetime.strptime('2019-02-01','%Y-%m-%d'):
        return 3
    elif dt_date>=datetime.datetime.strptime('2019-02-01','%Y-%m-%d') and dt_date<datetime.datetime.strptime('2019-08-01','%Y-%m-%d'):
        return 4
    elif dt_date>=datetime.datetime.strptime('2019-08-01','%Y-%m-%d') and dt_date<datetime.datetime.now():
        return 5
    else:
        raise ValueError("Wrong date %s" %str(dt_date))
        
#------------------------------------------------------------------------------
############################ DATA PREPROCESSING ###############################
#------------------------------------------------------------------------------
df_import = pd.read_csv(file_name)

# Process and add new features
df_import.loc[:,'category'] = df_import['category'].str.strip()
df_import.loc[:,'multipliers'] = df_import['metadata'].apply(lambda x: has_multiplier(x))
df_import.loc[:,'ad'] = df_import['metadata'].apply(lambda x: has_ad(x))
df_import.loc[:,'startActual'] = df_import['startActual'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z'))
df_import.loc[:,'tot_lives'] = df_import['tot_lives'].fillna(0)
df_import.loc[:,'prize'] = df_import['prizeCents'].apply(lambda x: float(x)/100.)
df_import.loc[:,'weekDay'] = df_import.loc[:,'startActual'].apply(lambda x: is_weekday(x.weekday()))
df_import.loc[:,'timeOfDay'] = df_import.loc[:,'startActual'].apply(lambda x: get_time_of_day(x))
df_import.loc[:,'period'] = df_import['startActual'].apply(lambda x: get_period(x))
df_import = df_import.sort_values(by=['startActual','q_order'])

# Select right answers, legitimate shows (with a non null starting audience) and questions that have non null total answers
df = df_import[(df_import['correct']==1.0) & (df_import['start_audience']>0) & (df_import['tot_users']>0)]

# Select legitimate shows (with at least some data on lives used)
df_nodata=df.groupby(['showId']).agg({'tot_lives':'sum'})
showId_nodata = df_nodata[df_nodata['tot_lives']==0].index.tolist()
showId_nodata.append(4817)
df = df[~(df['showId'].isin(showId_nodata))]

# Compute the actual last question number (with at least 1 answer)
showId_max_questions = df.groupby(['showId']).agg({'q_order':'max'})
df = df.join(showId_max_questions,on='showId',how='left',lsuffix='_1',rsuffix='_2')

df = df.rename(columns={"last_q": "old_last_q", "q_order_2": "total_questions"})

# Compute a few additional features
df.loc[:,'%_remaining_players'] = df['tot_users']/df['start_audience']
df.loc[:,'difficulty'] = 1. - df['count']/df['tot_users']
df.loc[:,'progress'] = df['q_order_1']/df['total_questions']
df.loc[:,'tot_lives_per_wrong_answer'] = df['tot_lives']/(df['tot_users']-df['count'])
df.loc[:,'tot_lives_per_wrong_answer'] = df.loc[:,'tot_lives_per_wrong_answer'].fillna(0)

# Remove unreasonnable values and remove last question of each show (no extra life allowed)
df = df[(df['tot_lives_per_wrong_answer']<=1.0)]
df = df[~(df['total_questions']==df['q_order_1'])]

# Categorical data hot encoding
df_final = pd.get_dummies(df, columns=['weekDay','timeOfDay','category','showType'], dummy_na=False, dtype='int64')

# Check NaN values
print('Number of nan values in each column')
print(df_final.isna().sum())
description=df_final.describe()

#------------------------------------------------------------------------------
#################################### EDA ######################################
#------------------------------------------------------------------------------
## Effect of question number
df_plot=df_final[df_final['total_questions']==12]
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(211)
sns.boxplot(x=df_plot['q_order_1'], y=df_plot['tot_lives'], ax=ax1, showmeans=True, meanline=True)
plt.xlim([-0.5,10.5])
plt.xlabel('')
plt.ylim(bottom=0)
plt.ylabel('Number of lives used')
plt.title('Distribution of lives used per question number for 12-round shows')
ax2 = fig1.add_subplot(212)
sns.boxplot(x=df_plot['q_order_1'], y=df_plot['tot_lives_per_wrong_answer'], ax=ax2, showmeans=True, meanline=True)
plt.xlim([-0.5,10.5])
plt.ylim(bottom=0)
plt.xlabel('Question number')
plt.ylabel('Number of lives used per wrong answer')

## Effect of question category
df_plot=df.groupby(['category']).mean()[['tot_lives', 'tot_lives_per_wrong_answer']]
fig2 = plt.figure(2)
ax21 = fig2.add_subplot(111)
df_plot.reset_index().plot.bar(x='category',y='tot_lives_per_wrong_answer',ax=ax21, legend=False)
ax21.set_xticklabels(ax21.get_xticklabels(), rotation=45)
plt.title('Distribution of lives used per wrong answer for different question categories')
plt.tight_layout()
plt.xlabel('')

## Effect of prize value...
# On general audience
df_plot = df_final[df_final['total_questions']==12].groupby(['showId']).agg(
                        {'prize':'mean','start_audience':'mean','tot_lives_per_wrong_answer':'mean','tot_lives':'sum','total_questions':'mean'})
fig3 = plt.figure(3)
ax31 = fig3.add_subplot(311)
df_plot.plot.scatter(x='prize',y='start_audience',ax=ax31)
plt.ylim(bottom=0)
ax31.set_xscale('log')
plt.xlabel('')
plt.ylabel('Starting audience')
plt.title('Effect of prize value alone for 12-round shows')

# On number of lives per wrong answer
ax32 = fig3.add_subplot(312)
df_plot.plot.scatter(x='prize',y='tot_lives_per_wrong_answer',ax=ax32)
#Y = np.reshape(df_plot['tot_lives_per_wrong_answer'].values,[df_plot.shape[0],1])
#X = np.reshape(df_plot['prize'].values,[df_plot.shape[0],1])
#reg = LinearRegression().fit(X, Y)
#a1, b1 = reg.coef_, reg.intercept_
#Y = a1*X + b1
#df_plot['test']=Y
#df_plot.plot(x='prize',y='test',ax=ax32,legend=False)
ax32.set_xscale('log')
plt.xlabel('')
plt.ylabel('Average number of lives \n per wrong answer')

# On total number of lives
ax33 = fig3.add_subplot(313)
df_plot.plot.scatter(x='prize',y='tot_lives',ax=ax33)
ax33.set_xscale('log')
plt.xlabel('Prize value')
plt.ylabel('Average total number \n of lives used')

# Effect of time on lives used per show
df_plot = df_final.groupby(['startActual']).mean()['tot_lives_per_wrong_answer']
fig33 = plt.figure(33)
ax333 = fig33.add_subplot(111)
plt.scatter(df_plot.index,df_plot)
#ax333.set_xticklabels(ax333.get_xticklabels(), rotation=45)
plt.xlabel('Time')
plt.ylabel('Average ratio of lives used per show')
plt.title('Evolution of target variable values over time')

#------------------------------------------------------------------------------
######################## FEATURE SIGNIFICANCE #################################
#------------------------------------------------------------------------------
# Select time window
df_final = df_final[(df_final['startActual']>='2018-01-01') & (df_final['startActual']<=datetime.datetime.now())]

# Remove outliers from shows' question #1
#Q1 = df_final[df_final['q_order_1']==1].loc[:,'tot_lives_per_wrong_answer'].quantile(0.25)
#Q3 = df_final[df_final['q_order_1']==1].loc[:,'tot_lives_per_wrong_answer'].quantile(0.75)
#IQR = Q3 - Q1
#showId_inconsistent = df_final[(df_final['q_order_1']==1) & (df_final['tot_lives_per_wrong_answer']>Q3+IQR)]['showId'].values
#df_final = df_final[~((df_final['showId'].isin(showId_inconsistent)) & (df_final['q_order_1']==1))]
df_final = df_final[~(df_final['q_order_1']==1)]

# Shuffle order of the samples
df_final = df_final.sample(frac=1).reset_index(drop=True)

# Remove unnecessary columns
df_final = df_final.drop(['showId','startActual','prizeCents','questionId','q_order_1',
                          'answerId','a_order','correct','count','tot_lives', 'old_last_q',
                          'metadata','weekDay_1','showType_hq-global'],axis=1)

# Create training and testing sets
X = df_final.loc[:,df_final.columns != 'tot_lives_per_wrong_answer']  #independent columns
y = df_final.loc[:,'tot_lives_per_wrong_answer']  #target column i.e num of lives per wrong answer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# 1. Univariate Selection
N_features = 8
bestfeatures = SelectKBest(score_func=f_regression, k=N_features) #apply SelectKBest class to extract top N best features
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print('')
print('Top %i best features using SelectKBest class' %N_features)
print(featureScores.nlargest(N_features,'Score'))  #print N best features
fig = plt.figure(22)
ax = fig.add_subplot(111)
featureScores.nlargest(N_features,'Score').plot(kind='barh',ax=ax,fontsize=7,legend=False)
plt.yticks(np.arange(0,N_features),labels=featureScores.nlargest(N_features,'Score')['Specs'].values)
plt.xlabel('Score')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(10)

## 2.Correlation Matrix with Heatmap
corrmat = df_final.corr()
top_corr_features = corrmat.index
fig4 = plt.figure(4)
ax41 = fig4.add_subplot(111)
num = np.arange(1,len(df_final.columns)+1)
g=sns.heatmap(df_final[top_corr_features].corr(), ax=ax41,annot=False,cmap="RdYlGn",square=True,xticklabels=num,yticklabels=True)
ax41.set_xticklabels(ax41.get_xticklabels(), rotation=0)
plt.title('Correlation matrix of question features')

# 3. Feature Importance (using Random Forests and using Extremely Randomized Trees)
fig5 = plt.figure(5)
ax51 = fig5.add_subplot(121)
#gsc = GridSearchCV(
#        estimator=ExtraTreesRegressor(),
#        param_grid={
#            'max_depth': range(5,20),
#            'n_estimators': (20, 50, 100, 200),},
#        cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1, return_train_score=True)  # Look for best parameters for ExtraTreesRegressor
#etr = gsc.fit(X_train, y_train)
#best_params_etr = etr.best_params_
#best_etr_model = ExtraTreesRegressor(max_depth=best_params_etr["max_depth"], n_estimators=best_params_etr["n_estimators"])
#scores_etr = cross_val_score(best_etr_model, X, y, cv=10, scoring='neg_mean_absolute_error')
#best_etr_model.fit(X_train,y_train)
#print('')
#print('scores ETR')
#print(scores_etr)
#print('scores ETR on holdout set')
#print(mean_absolute_error(best_etr_model.predict(X_test),y_test))
#
#feat_importances_etr = pd.Series(best_etr_model.feature_importances_, index=X.columns)
#feat_importances_etr.nlargest(N_features).plot(kind='barh',ax=ax51,fontsize=7)
#plt.title('Using ExtraTreesRegressor', fontsize=11)
#plt.xlabel('Feature importance')
#ax51.set_yticklabels(ax51.get_yticklabels(), rotation=45)

ax52 = fig5.add_subplot(122)
gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(10,20),
            'n_estimators': (10, 20, 50, 100, 200, 400),},
        cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1, return_train_score=True)  # Look for best parameters for RandomForestRegressor
rfr = gsc.fit(X_train, y_train)
best_params_rfr = rfr.best_params_
best_rfr_model = RandomForestRegressor(max_depth=best_params_rfr["max_depth"], n_estimators=best_params_rfr["n_estimators"])
scores_rfr = cross_val_score(best_rfr_model, X, y, cv=10, scoring='neg_mean_absolute_error')
best_rfr_model.fit(X_train,y_train)
print('')
print('scores RFR with cross validation')
print(scores_rfr)
print('scores RFR on holdout set')
print(mean_absolute_error(best_rfr_model.predict(X_test),y_test))

feat_importances_rfr = pd.Series(best_rfr_model.feature_importances_, index=X.columns)
feat_importances_rfr.nlargest(N_features).plot(kind='barh',ax=ax52,fontsize=7)
plt.title('Using RandomForestRegressor', fontsize=11)
plt.xlabel('Feature importance')
ax52.set_yticklabels(ax52.get_yticklabels(), rotation=45)

export_graphviz(best_rfr_model.estimators_[0],out_file='tree.dot',
                feature_names=X_train.columns,
                filled=True,
                rounded=True)
##(insight) Antoines-MacBook-Pro:Main antoinepetit$ dot -Tpng tree.dot -o tree.png
#from IPython.display import Image
#Image(filename = 'tree.png')

#------------------------------------------------------------------------------
############### PREDICTION USING LINEAR RIDGE REGRESSION #####################
#------------------------------------------------------------------------------
#gsc = GridSearchCV(estimator=ElasticNet(normalize=True),
#                   param_grid={'alpha':np.logspace(-5,2,8),
#                               'l1_ratio':[.2,.4,.6,.8]},
#                               cv=5, scoring='neg_mean_squared_error',n_jobs=1,refit=True)
#lm_elastic = gsc.fit(X_train, y_train)
#best_params_elastic = lm_elastic.best_params_
#best_lm_elastic_model = ElasticNet(normalize=True,alpha=best_params_elastic["alpha"], l1_ratio=best_params_elastic["l1_ratio"])
#best_lm_elastic_model.fit(X_train,y_train)
#
#print(best_lm_elastic_model.score(X_test, y_test))
#------------------------------------------------------------------------------
####################### SCENARIO-BASED OPTIMIZATION ###########################
#------------------------------------------------------------------------------
df_edit = df_final.copy(deep=True)
X = df_edit.loc[:,df_edit.columns != 'tot_lives_per_wrong_answer']  #independent columns
y = df_edit.loc[:,'tot_lives_per_wrong_answer']  #target column
best_rfr_model.fit(X,y)

# Scenario parameters
period = 5
rounds = 12
last_q = rounds
prize = 2000 # USD
start_audience = 100000
target_final_audience = 100
multipliers = 1.0
ad = 0.0
weekDay_0 = 0
timeOfDay = np.array([[1,0,0,0]])
showType = np.array([[1,0]])
category = np.zeros([1,21])
scenarios = np.array([[0.06,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7],
                      [0.505,0.75,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,0.72],
                      [0.04,	0.35,0.55,0.65,0.75,0.1,0.2,0.3,0.4,0.5,0.73],
                      [0.17,	0.35,0.58,0.12,0.35,0.55,0.7,0.1,0.26,0.59,0.79],
                      [0.465,0.465,0.465,0.465,0.465,0.465,0.465,0.465,0.465,0.465,0.465]])

# Simulation
audiences = [[] for _ in range(len(scenarios))]
for sid,difficulty in enumerate(scenarios):
    print('Scenario %i'%sid)
    tot_lives_purchased = 0.0
    curr_audience = start_audience
    audiences[sid].append(curr_audience)
    for rid in np.arange(0,rounds-1):
        wrong_answers = (1-difficulty[rid])*curr_audience
        data = np.concatenate((np.array([[curr_audience,start_audience,multipliers,ad,prize,period,
                                          last_q,curr_audience/start_audience,difficulty[rid],
                                          float(rid+1)/rounds,weekDay_0]]),
                                timeOfDay,category, showType),axis=1)
        extra_lives = best_rfr_model.predict(data)[0]
        curr_audience = curr_audience + extra_lives*wrong_answers - wrong_answers
        tot_lives_purchased += extra_lives*wrong_answers
        audiences[sid].append(curr_audience)        
    print(tot_lives_purchased)
    print(curr_audience)
