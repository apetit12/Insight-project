#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:11:57 2019

@author: antoinepetit
"""

file_name = 'bq-results-20190930-002010-questions.csv'

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, SCORERS
import statsmodels.api as sm

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
    
#------------------------------------------------------------------------------
############################ DATA PREPROCESSING ###############################
#------------------------------------------------------------------------------
df_import = pd.read_csv(file_name)

# Process and add a few features
df_import.loc[:,'category'] = df_import['category'].str.strip()
df_import.loc[:,'multipliers'] = df_import['metadata'].apply(lambda x: has_multiplier(x))
df_import.loc[:,'ad'] = df_import['metadata'].apply(lambda x: has_ad(x))
df_import.loc[:,'startActual'] = df_import['startActual'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z'))
df_import.loc[:,'tot_lives'] = df_import['tot_lives'].fillna(0)
df_import.loc[:,'prize'] = df_import['prizeCents'].apply(lambda x: float(x)/100)
df_import.loc[:,'weekDay'] = df_import.loc[:,'startActual'].apply(lambda x: is_weekday(x.weekday()))
df_import = df_import.sort_values(by=['startActual','q_order'])

# Select time window
df = df_import[(df_import['startActual']>='2018-01-01') & (df_import['startActual']<datetime.datetime.now())]

# Select right answers, legitimate shows (with a non null starting audience) and questions that have answers
df = df[(df['correct']==1.0) & (df['start_audience']>0) & (df['tot_users']>0)]

# Select legitimate shows (with at least some data on lives used)
df_nodata=df.groupby(['showId']).agg({'tot_lives':'sum'})
showId_nodata = df_nodata[df_nodata['tot_lives']==0].index.tolist()
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
df_final = pd.get_dummies(df, columns=['weekDay','category','showType'], dummy_na=False, dtype='int64')

# Check NaN values
print('Number of nan values in each column')
print(df_final.isna().sum())
description=df_final.describe()

#------------------------------------------------------------------------------
#################################### EDA ######################################
#------------------------------------------------------------------------------
### Effect of question number
#df_plot=df_final[df_final['last_q']==12]
#fig1 = plt.figure(1)
#ax1 = fig1.add_subplot(211)
#sns.boxplot(x=df_final['q_order'], y=df_plot['tot_lives'], ax=ax1, showmeans=True, meanline=True)
#plt.xlim([-0.5,10.5])
#plt.xlabel('')
#plt.ylim(bottom=0)
#plt.ylabel('number of lives used')
#plt.title('Distribution of lives used per question number for 12-round shows')
#ax2 = fig1.add_subplot(212)
#sns.boxplot(x=df_final['q_order'], y=df_plot['tot_lives_per_wrong_answer'], ax=ax2, showmeans=True, meanline=True)
#plt.xlim([-0.5,10.5])
#plt.ylim(bottom=0)
#plt.xlabel('question number')
#plt.ylabel('number of lives used per wrong answer')
#
### Effect of question category
#df_plot=df.groupby(['category']).mean()[['tot_lives', 'tot_lives_per_wrong_answer']]
#fig2 = plt.figure(2)
#ax21 = fig2.add_subplot(111)
#df_plot.reset_index().plot.bar(x='category',y='tot_lives_per_wrong_answer',ax=ax21, legend=False)
#ax21.set_xticklabels(ax21.get_xticklabels(), rotation=45)
#plt.title('Distribution of lives used per wrong answer for different question categories')
#plt.tight_layout()
#plt.xlabel('')
#
### Effect of prize value...
## On general audience
#df_plot = df[(df['startActual']>='2018-01-01') & (df['startActual']<='2019-01-01') & (df_final['last_q']==12)].groupby(['showId']).agg(
#                        {'prizeCents':'mean','start_audience':'mean','tot_lives_per_wrong_answer':'mean','tot_lives':'sum','last_q':'mean'})
#fig3 = plt.figure(3)
#ax31 = fig3.add_subplot(311)
#df_plot.plot.scatter(x='prizeCents',y='start_audience',ax=ax31)
##plt.xlim([0,45000000])
#plt.ylim(bottom=0)
#ax31.set_xscale('log')
#plt.xlabel('')
#plt.ylabel('Starting audience')
#plt.title('Effect of prize value alone for 12-round shows')
#
## On number of lives per wrong answer
#ax32 = fig3.add_subplot(312)
#df_plot.plot.scatter(x='prizeCents',y='tot_lives_per_wrong_answer',ax=ax32)
##Y = np.reshape(df_plot['tot_lives_per_wrong_answer'].values,[df_plot.shape[0],1])
##X = np.reshape(df_plot['prizeCents'].values,[df_plot.shape[0],1])
##reg = LinearRegression().fit(X, Y)
##a1, b1 = reg.coef_, reg.intercept_
##Y = a1*X + b1
##df_plot['test']=Y
##df_plot.plot(x='prizeCents',y='test',ax=ax32,legend=False)
##plt.xlim([0,45000000])
#ax32.set_xscale('log')
#plt.xlabel('')
#plt.ylabel('Average number of lives \n per wrong answer')
#
## On total number of lives
#ax33 = fig3.add_subplot(313)
#df_plot.plot.scatter(x='prizeCents',y='tot_lives',ax=ax33)
##Y = np.reshape(df_plot['tot_lives'].values,[df_plot.shape[0],1])
##X = np.reshape(df_plot['prizeCents'].values,[df_plot.shape[0],1])
##reg = LinearRegression().fit(X, Y)
##a2, b2 = reg.coef_, reg.intercept_
##Y = a2*X + b2
##df_plot['test2']=Y
##df_plot.plot(x='prizeCents',y='test2',ax=ax33,legend=False)
##plt.xlim([0,45000000])
#ax33.set_xscale('log')
#plt.xlabel('Prize value (cents)')
#plt.ylabel('Average total number \n of lives used')
#
#------------------------------------------------------------------------------
######################## FEATURE SIGNIFICANCE #################################
#------------------------------------------------------------------------------
df_final = df_final.drop(['showId','startActual','prizeCents','questionId','q_order_1',
                          'answerId','a_order','correct','count','tot_lives', 'old_last_q',
                          'start_audience','metadata','weekDay_1','showType_hq-global'],axis=1)
X = df_final.loc[:,df_final.columns != 'tot_lives_per_wrong_answer']  #independent columns
y = df_final.loc[:,'tot_lives_per_wrong_answer']    #target column i.e price range

## 1. Univariate Selection
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

df_final = df_final.drop(['progress'],axis=1)
X = df_final.loc[:,df_final.columns != 'tot_lives_per_wrong_answer']  #independent columns
y = df_final.loc[:,'tot_lives_per_wrong_answer']    #target column i.e price range

## 3. Feature Importance (using Random Forests and using Extremely Randomized Trees)
#print(sorted(SCORERS.keys()))  # for all scorer options
fig5 = plt.figure(5)
ax51 = fig5.add_subplot(131)
#gsc = GridSearchCV(
#        estimator=ExtraTreesRegressor(),
#        param_grid={
#            'max_depth': range(3,8),
#            'n_estimators': (10, 20, 50, 100, 200),},
#        cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)  # Look for best parameters for ExtraTreesRegressor
#
#best_params_etr = gsc.fit(X, y).best_params_
#best_etr_model = ExtraTreesRegressor(max_depth=best_params_etr["max_depth"], n_estimators=best_params_etr["n_estimators"])
#scores_etr = cross_val_score(best_etr_model, X, y, cv=10, scoring='neg_mean_absolute_error')
#best_etr_model.fit(X,y)
#print('')
#print('Top %i most important features using ExtraTreesRegressor' %N_features)
#print(best_etr_model.feature_importances_) #use inbuilt class feature_importances of tree based regressor
#print('scores ETR')
#print(scores_etr)
#feat_importances_etr = pd.Series(best_etr_model.feature_importances_, index=X.columns)
#feat_importances_etr.nlargest(N_features).plot(kind='barh',ax=ax51,fontsize=7)
#plt.title('Using ExtraTreesRegressor', fontsize=11)
#plt.xlabel('Feature importance')
#ax51.set_yticklabels(ax51.get_yticklabels(), rotation=45)
#
ax52 = fig5.add_subplot(132)
gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,8),
            'n_estimators': (10, 20, 50, 100, 200),},
        cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)  # Look for best parameters for RandomForestRegressor

best_params_rfr = gsc.fit(X, y).best_params_
best_rfr_model = RandomForestRegressor(max_depth=best_params_rfr["max_depth"], n_estimators=best_params_rfr["n_estimators"])
scores_rfr = cross_val_score(best_rfr_model, X, y, cv=10, scoring='neg_mean_absolute_error')
best_rfr_model.fit(X,y)
print('')
print('Top %i most important features using RandomForestRegressor' %N_features)
print(best_rfr_model.feature_importances_) #use inbuilt class feature_importances of tree based regressor
print('scores RFR')
print(scores_rfr)
feat_importances_rfr = pd.Series(best_rfr_model.feature_importances_, index=X.columns)
feat_importances_rfr.nlargest(N_features).plot(kind='barh',ax=ax52,fontsize=7)
plt.title('Using RandomForestRegressor', fontsize=11)
plt.xlabel('Feature importance')
ax52.set_yticklabels(ax52.get_yticklabels(), rotation=45)

#ax53 = fig5.add_subplot(133)
#test = pd.concat((feat_importances_rfr,feat_importances_etr),axis=1).mean(axis=1).nlargest(N_features).plot(kind='barh',ax=ax53,fontsize=7)   
#plt.title('Mean value', fontsize=11)
#plt.xlabel('Average feature importance')
#ax53.set_yticklabels(ax53.get_yticklabels(), rotation=45)
#
#plt.suptitle('Top %i most important features' %N_features)
##plt.tight_layout(rect=[0, 0.03, 1, 0.9])
#
##------------------------------------------------------------------------------
##################### PREDICTION USING LINEAR REGRESSION #######################
##------------------------------------------------------------------------------
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
#ols = sm.OLS(y_train, sm.add_constant(X_train))
#ols2 = ols.fit()
#print(ols2.summary())
#y_pred = ols2.predict(sm.add_constant(X_test))
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
#print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
#
### Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test, y_pred))
#
##------------------------------------------------------------------------------
######################## SCENARIO-BASED OPTIMIZATION ###########################
##------------------------------------------------------------------------------
#X = df_opt.loc[:,df_opt.columns != 'tot_lives_per_wrong_answer']  #independent columns
#y = df_opt.loc[:,'tot_lives_per_wrong_answer']    #target column i.e price range
#best_etr_model.fit(X,y)
#best_rfr_model.fit(X,y)
#
#rounds = 12
##last_q = rounds*np.ones([1,rounds-1])
#last_q = rounds
##prizeCents = 200000*np.ones([1,rounds-1]) # cents
#prizeCents = 200000 # cents
#start_audience = 100000
#target_final_audience = 100
##multipliers = 1.0*np.ones([1,rounds-1])
#multipliers = 1.0
##ad = 0.0*np.ones([1,rounds-1])
#ad = 0.0
##showType = np.transpose(np.tile(np.array([1,0,0]),(rounds-1,1)))
#showType = np.array([[1,0,0]])
##progress = np.reshape(1./12*np.arange(1,12),[1,rounds-1])
##scenarios = np.array([[1.,0.495,	0.25,0.9,0.85,0.8,0.75,0.7,0.6,0.5,0.3,0.28],
##             [1.,0.96,0.65,0.45,0.35,0.25,0.9,0.8,0.7,0.6,0.5,0.27],
##             [1.,0.94,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3]])
#scenarios = np.array([[0.495,	0.25,0.9,0.85,0.8,0.75,0.7,0.6,0.5,0.3,0.28],
#             [0.96,0.65,0.45,0.35,0.25,0.9,0.8,0.7,0.6,0.5,0.27],
#             [0.94,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3],
#             [0.83,0.65,	0.42,0.88,0.65,0.45,0.3,0.9,0.74,0.41,0.21]])
#
##tot_users = [np.reshape(start_audience*np.cumprod(scenario),[1,rounds]) for scenario in scenarios]
##remaining_players = [np.reshape(np.cumprod(scenario),[1,rounds]) for scenario in scenarios]
#category = np.zeros([1,21])
#
#for sid,scenario in enumerate(scenarios):
#    print('Scenario %i'%sid)
##    print('Scenario %i:' %sid)
##    tot_users = np.reshape(start_audience*np.cumprod(scenario[:-1]),[1,rounds-1])
##    remaining_players = np.reshape(np.cumprod(scenario[:-1]),[1,rounds-1])
##
##    data = np.transpose(np.concatenate((prizeCents,tot_users,last_q,multipliers,ad,
##                                        np.reshape(scenario[1:],[1,rounds-1]),remaining_players,category,showType),axis=0))   
##    
##    audience_left = np.reshape(start_audience*np.cumprod(scenario),[1,rounds])
##    audience_eliminated = audience_left[0,:-1]-audience_left[0,1:]
##    print(audience_eliminated)
##    print(best_etr_model.predict(data))
##    rev_1 = np.sum(np.multiply(audience_eliminated,best_etr_model.predict(data)))
##    rev_2 = np.sum(np.multiply(audience_eliminated,best_rfr_model.predict(data)))
##    print((rev_1+rev_2)/2)
#    
#    tot_lives_purchased = 0.0
#    curr_audience = start_audience
#    for rid in np.arange(0,11):
#        wrong_answers = (1-scenario[rid])*curr_audience
#        data = np.concatenate((np.array([[prizeCents,curr_audience,last_q,multipliers,ad,scenario[rid],curr_audience/start_audience]]),
#                                                                               category, showType),axis=1)
#        extra_lives = best_rfr_model.predict(data)[0]
#        curr_audience = curr_audience + extra_lives*wrong_answers - wrong_answers
#        tot_lives_purchased += extra_lives*wrong_answers
#    print(tot_lives_purchased)
#    print(curr_audience)
