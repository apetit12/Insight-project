# -*- coding: utf-8 -*-
file_name = 'bq-results-20190921-175210-questions.csv'

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
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

#------------------------------------------------------------------------------
################################ FUNCTIONS ####################################
#------------------------------------------------------------------------------
def has_ad(metadata):
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

def generate_table(dataframe, max_rows=30, max_cols=8):
    return html.Table(
        # Header
        [html.Tr([html.Th(col, style={'color': colors['text']}) for col in dataframe.columns[:max_cols]])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col], style={'color': colors['text']}) for col in dataframe.columns[:max_cols]
        ]) for i in range(min(len(dataframe), max_rows))]
    )

#------------------------------------------------------------------------------
############################ DATA PREPROCESSING ###############################
#------------------------------------------------------------------------------
df_import = pd.read_csv(file_name)
df_import.loc[:,'category'] = df_import['category'].str.strip()
df_import.loc[:,'multipliers'] = df_import['metadata'].apply(lambda x: has_multiplier(x))
df_import.loc[:,'ad'] = df_import['metadata'].apply(lambda x: has_ad(x))
df_import.loc[:,'startTime'] = df_import['startTime'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z'))
df_import.loc[:,'tot_lives'] = df_import['tot_lives'].fillna(0)
df_import.loc[:,'progress'] = df_import['order']/df_import['last_q']
df_import.loc[:,'difficulty'] = df_import['count']/df_import['tot_users']
df_import.loc[:,'remaining_players'] = df_import['tot_users']/df_import['start_audience']
df_import.loc[:,'tot_lives_per_wrong_answer'] = df_import['tot_lives']/(df_import['tot_users']-df_import['count'])
df_import.loc[:,'tot_lives_per_wrong_answer'] = df_import.loc[:,'tot_lives_per_wrong_answer'].fillna(0)

df_import = df_import.sort_values(by=['startTime','order'])

#------------------------------------------------------------------------------
############################## DATA SELECTION #################################
#------------------------------------------------------------------------------
df = df_import[(df_import['startTime']>='2018-01-01') & (df_import['startTime']<='2020-01-01') 
                        & (df_import['correct']==1.0) & (df_import['tot_lives_per_wrong_answer']<=1.0)
                        & (df_import['start_audience']>0) & (df_import['tot_users']>0)]
df_final = pd.get_dummies(df, columns=['category','showType'], dummy_na=False, dtype='int64')
df_final = df_final.sort_values(by=['startTime','order'])

description=df_final.describe()

#------------------------------------------------------------------------------
#################################### EDA ######################################
#------------------------------------------------------------------------------
## Effect of question number
df_plot=df_final[df_final['last_q']==12]
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(211)
# sns.boxplot(x=df_final['order'], y=df_plot['tot_lives'], ax=ax1, showmeans=True, meanline=True)
# plt.xlim([-0.5,10.5])
# plt.xlabel('')
# plt.ylim(bottom=0)
# plt.ylabel('number of lives used')
# plt.title('Distribution of lives used per question number for 12-round shows')
# ax2 = fig1.add_subplot(212)
# sns.boxplot(x=df_final['order'], y=df_plot['tot_lives_per_wrong_answer'], ax=ax2, showmeans=True, meanline=True)
# plt.xlim([-0.5,10.5])
# plt.ylim(bottom=0)
# plt.xlabel('question number')
# plt.ylabel('number of lives used per wrong answer')

## Effect of question category
df_plot2=df.groupby(['category']).mean()[['tot_lives', 'tot_lives_per_wrong_answer']]
# fig2 = plt.figure(2)
# ax21 = fig2.add_subplot(111)
# df_plot2.reset_index().plot.bar(x='category',y='tot_lives_per_wrong_answer',ax=ax21, legend=False)
# ax21.set_xticklabels(ax21.get_xticklabels(), rotation=45)
# plt.title('Distribution of lives used per wrong answer for different question categories')
# plt.tight_layout()
# plt.xlabel('')

## Effect of prize value...
# On general audience
df_plot3 = df[(df['startTime']>='2018-01-01') & (df['startTime']<='2019-01-01') & (df_final['last_q']==12)].groupby(['showId']).agg(
                        {'prizeCents':'mean','start_audience':'mean','tot_lives_per_wrong_answer':'mean','tot_lives':'sum','last_q':'mean'})
# fig3 = plt.figure(3)
# ax31 = fig3.add_subplot(311)
# df_plot3.plot.scatter(x='prizeCents',y='start_audience',ax=ax31)
# plt.ylim(bottom=0)
# ax31.set_xscale('log')
# plt.xlabel('')
# plt.ylabel('Starting audience')
# plt.title('Effect of prize value alone for 12-round shows')

# On number of lives per wrong answer
# ax32 = fig3.add_subplot(312)
# df_plot3.plot.scatter(x='prizeCents',y='tot_lives_per_wrong_answer',ax=ax32)
# ax32.set_xscale('log')
# plt.xlabel('')
# plt.ylabel('Average number of lives \n per wrong answer')

# On total number of lives
# ax33 = fig3.add_subplot(313)
# df_plot3.plot.scatter(x='prizeCents',y='tot_lives',ax=ax33)
# ax33.set_xscale('log')
# plt.xlabel('Prize value (cents)')
# plt.ylabel('Average total number \n of lives used')

#------------------------------------------------------------------------------
######################## FEATURE SIGNIFICANCE #################################
#------------------------------------------------------------------------------
df_final = df_final.drop(['showId','questionId','order','count','correct','tot_lives',
                          'startTime','start_audience','metadata','answerId'],axis=1)

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

## 2.Correlation Matrix with Heatmap
corrmat = df_final.corr()
top_corr_features = corrmat.index
# fig4 = plt.figure(4)
# ax41 = fig4.add_subplot(111)
# num = np.arange(1,len(df_final.columns)+1)
# g=sns.heatmap(df_final[top_corr_features].corr(), ax=ax41,annot=False,cmap="RdYlGn",square=True,xticklabels=num,yticklabels=True)
# ax41.set_xticklabels(ax41.get_xticklabels(), rotation=0)
# plt.title('Correlation matrix of question features')

## 3. Feature Importance (using Random Forests and using Extremely Randomized Trees)
#fig5 = plt.figure(5)
#ax51 = fig5.add_subplot(131)
gsc = GridSearchCV(
        estimator=ExtraTreesRegressor(),
        param_grid={
            'max_depth': range(3,8),
            'n_estimators': (10, 20, 50, 100, 200),},
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)  # Look for best parameters for ExtraTreesRegressor

best_params_etr = gsc.fit(X, y).best_params_
best_etr_model = ExtraTreesRegressor(max_depth=best_params_etr["max_depth"], n_estimators=best_params_etr["n_estimators"])
scores_etr = cross_val_score(best_etr_model, X, y, cv=10, scoring='neg_mean_absolute_error')
best_etr_model.fit(X,y)
#print('')
#print('Top %i most important features using ExtraTreesRegressor' %N_features)
#print(best_etr_model.feature_importances_) #use inbuilt class feature_importances of tree based regressor
#print('scores ETR')
#print(scores_etr)
feat_importances_etr = pd.Series(best_etr_model.feature_importances_, index=X.columns)
#feat_importances_etr.nlargest(N_features).plot(kind='barh',ax=ax51,fontsize=7)
#plt.title('Using ExtraTreesRegressor', fontsize=11)
#plt.xlabel('Feature importance')
#ax51.set_yticklabels(ax51.get_yticklabels(), rotation=45)

#ax52 = fig5.add_subplot(132)
gsc = GridSearchCV(
       estimator=RandomForestRegressor(),
       param_grid={
           'max_depth': range(3,8),
            'n_estimators': (10, 20, 50, 100, 200),},
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)  # Look for best parameters for RandomForestRegressor

best_params_rfr = gsc.fit(X, y).best_params_
best_rfr_model = RandomForestRegressor(max_depth=best_params_rfr["max_depth"], n_estimators=best_params_rfr["n_estimators"])
scores_rfr = cross_val_score(best_rfr_model, X, y, cv=10, scoring='neg_mean_absolute_error')
best_rfr_model.fit(X,y)
#print('')
#print('Top %i most important features using RandomForestRegressor' %N_features)
#print(best_rfr_model.feature_importances_) #use inbuilt class feature_importances of tree based regressor
#print('scores RFR')
#print(scores_rfr)
feat_importances_rfr = pd.Series(best_rfr_model.feature_importances_, index=X.columns)
#feat_importances_rfr.nlargest(N_features).plot(kind='barh',ax=ax52,fontsize=7)
#plt.title('Using RandomForestRegressor', fontsize=11)
#plt.xlabel('Feature importance')
#ax52.set_yticklabels(ax52.get_yticklabels(), rotation=45)

#ax53 = fig5.add_subplot(133)
#test = pd.concat((feat_importances_rfr,feat_importances_etr),axis=1).mean(axis=1).nlargest(N_features).plot(kind='barh',ax=ax53,fontsize=7)   
#plt.title('Mean value', fontsize=11)
#plt.xlabel('Average feature importance')
#ax53.set_yticklabels(ax53.get_yticklabels(), rotation=45)
#
#plt.suptitle('Top %i most important features' %N_features)
#plt.tight_layout(rect=[0, 0.03, 1, 0.9])

q_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
scenario_A = [0.505, 0.75, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.72]
scenario_B = [0.04, 0.35, 0.55, 0.65, 0.75, 0.1, 0.2, 0.3, 0.4, 0.5, 0.73]
scenario_C = [0.06, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
scenario_D = [0.17, 0.35, 0.58, 0.12, 0.35, 0.55, 0.7, 0.1, 0.26, 0.59, 0.79]

optim_results = [['Number of lives purchased',10723, 11616, 11873, 12241],['Remaining players at q=11',540, 563, 520, 585]]


#------------------------------------------------------------------------------
################################# DASH OBJECT #################################
#------------------------------------------------------------------------------
marginVal = 200
# Call an external CSS style sheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create the Dash object
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.css.append_css({'external_url': '/static/reset.css'})

#Page parameters
page_title = 'HQ Trivia'
app_description = "Understanding why users buy back in"
markdown_title = '''Project Summary'''
markdown_text_1 = '''
---
HQ Trivia is a popular mobile game developed by Intermedia Labs with over 150,000 daily 
users competing to earn a prize of $1,000+. Recently, Intermedia decided to switch to a 
different business model and are looking to diversify their revenue sources. In 
consultation with Intermedia, I designed a dashboard that identifies the most predictive 
features driving users to make in-game purchases. It will empower the editorial board to 
leverage these purchases as an additional revenue stream. Using multiple datasets from 
Intermedia’s extensive database, I devised customized features and used ensemble learning 
methods to quantify their importance.
'''
markdown_title_2 = "Dataset Summary"
markdown_text_2 = '''
## About the author
My name is Antoine Petit
'''
test1 = '''
---
Effect of question number
'''
test2 = '''
---
Effect of category
'''
test3 = '''
---
Effect of prize value (log-scale)
'''
test4 = '''
---
Univariate selection
'''
test5 = '''
---
Feature importance
'''
test6 = '''
---
Scenario-based game flow optimization
'''

colors = {
    'background': '#4B0082',
    'titles': '#f1c232',
    'text': '#f3f2f9'}

app.layout = html.Div(children=[
	
	# To generate <h1> components
    html.H1(children=page_title,
    	style={'textAlign': 'center', 'color': colors['titles'],'font-weight': 'bold','font-size': 36}
            ),

    html.Div(children=app_description,
    	style={'font-size': 24, 'textAlign': 'center','color': colors['text'],'padding': 30}
            ),

    html.Div(children=markdown_title,
    	style={'textAlign': 'center','color': colors['titles'],'font-weight': 'bold','font-size': 24}
            ),

    html.Div(children=[dcc.Markdown(children=markdown_text_1)],
    	style={'color': colors['text'], 'marginLeft': marginVal, 'marginRight': marginVal}
    		),

    html.Div(children=markdown_title_2,
        style={'textAlign': 'center','color': colors['titles']}
            ),

    html.Div(children=[
    	html.H4(children='Table title', 
    		    style={'color': colors['text']}),
    	        generate_table(description)
        	],
            style={'marginLeft': marginVal}
			),

    html.Div(children=[dcc.Markdown(children=test1)],
        style={'textAlign': 'center','color': colors['titles'], 'marginLeft': marginVal, 'marginRight': marginVal,'font-size': 24,'font-weight': 'bold'}
            ),

    dcc.Graph(
        id='graph1',
        figure={
            'data': [go.Box( y=df_plot[df_plot['order']==q_number]['tot_lives_per_wrong_answer'], name=str(q_number), showlegend=False) for q_number in df_plot.order.unique()
            ],
            'layout':
                {
#                    'title':'Comparison of different user types',
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'title':'Question number', 'color':colors['text'], 'tickvals':q_numbers},
                    'yaxis':{'title':'Number of lives used per wrong answer', 'color':colors['text']},
                    'hovermode':'closest'
                }
        	}
    	),

    html.Div(children=[dcc.Markdown(children=test2)],
        style={'textAlign': 'center','color': colors['titles'], 'marginLeft': marginVal, 'marginRight': marginVal,'font-size': 24,'font-weight': 'bold'}
            ),

    dcc.Graph(
        id='graph2',
        figure={
            'data': [go.Bar(x=df_plot2.reset_index()['category'],y=df_plot2.reset_index()['tot_lives_per_wrong_answer'], marker={
        'color': df_plot2.reset_index()['tot_lives_per_wrong_answer'],
        'colorscale': 'Viridis'})
            ],
            'layout':
                {
#                    'title':'Comparison of different user types',
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'title':'Question category', 'color':colors['text']},
                    'yaxis':{'title':'Number of lives used per wrong answer', 'color':colors['text']},
                }
            }
        ),

    html.Div(children=[dcc.Markdown(children=test3)],
        style={'textAlign': 'center','color': colors['titles'], 'marginLeft': marginVal, 'marginRight': marginVal,'font-size': 24,'font-weight': 'bold'}
            ),

    dcc.Graph(
        id='graph3',
        figure={
            'data': [
                {
                    'x': df_plot3['prizeCents'].values,
                    'y': df_plot3['start_audience'].values,
                    'mode': 'markers',
                    'marker': {'size': 12}
                },
            ],
            'layout':
                {
#                    'title':'Comparison of different user types',
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'title':'Prize value (cents)', 'color':colors['text'], 'type':"log", 'showgrid': False},
                    'yaxis':{'title':'Starting audience', 'color':colors['text']},
                }
            }
        ),


    dcc.Graph(
        id='graph4',
        figure={'data': [
                {
                    'x': df_plot3['prizeCents'].values,
                    'y': df_plot3['tot_lives_per_wrong_answer'].values,
                    'mode': 'markers',
                    'marker': {'size': 12}
                },
            ],
            'layout':
                {
#                    'title':'Comparison of different user types',
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'title':'Prize value (cents)', 'color':colors['text'], 'type':"log", 'showgrid': False},
                    'yaxis':{'title':'Number of lives used per wrong answer', 'color':colors['text']},
                }
            }
        ),

    html.Div(children=[dcc.Markdown(children=test4)],
        style={'textAlign': 'center','color': colors['titles'], 'marginLeft': marginVal, 'marginRight': marginVal,'font-size': 24,'font-weight': 'bold'}
            ),

    dcc.Graph(
        id='graph5',
        figure={
            'data': [go.Bar(y=featureScores.nlargest(N_features,'Score')['Specs'],x=featureScores.nlargest(N_features,'Score')['Score'], orientation='h',
                marker={'color': featureScores.nlargest(N_features,'Score')['Score'],'colorscale': 'Viridis'})
            ],
            'layout':
                {
#                    'title':'Comparison of different user types',
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'title':'Score', 'color':colors['text']},
                    'yaxis':{'color':colors['text'], 'automargin': True},
                }
            },
        style={'marginLeft': marginVal}

        ),

    dcc.Graph(
        id='graph6',
        figure={
            'data': [go.Heatmap(z=df_final.corr().values.tolist(), x=df_final.columns, y=df_final.columns, colorscale='ylgnbu')],
            'layout':
                {
#                    'title':'Comparison of different user types',
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'color':colors['text'], 'tickangle':45, 'automargin': True},
                    'yaxis':{'color':colors['text'], 'automargin': True},
                }
            },
        style={'marginLeft': marginVal}

        ),

    html.Div(children=[dcc.Markdown(children=test5)],
        style={'textAlign': 'center','color': colors['titles'], 'marginLeft': marginVal, 'marginRight': marginVal,'font-size': 24,'font-weight': 'bold'}
            ),

    dcc.Graph(
        id='graph7',
        figure={
            'data': [go.Bar(y=feat_importances_etr.nlargest(N_features).index,x=feat_importances_etr.nlargest(N_features), orientation='h',
            marker={'color': feat_importances_etr.nlargest(N_features),'colorscale': 'Viridis'})
            ],
            'layout':
                {
#                    'title':'Comparison of different user types',
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'title':'Score (ExtraTreesRegressor)', 'color':colors['text']},
                    'yaxis':{'color':colors['text'], 'automargin': True},
                }
            },
        style={'marginLeft': marginVal}

        ),

    dcc.Graph(
        id='graph8',
        figure={
            'data': [go.Bar(y=feat_importances_rfr.nlargest(N_features).index,x=feat_importances_rfr.nlargest(N_features), orientation='h',
            marker={'color': feat_importances_rfr.nlargest(N_features),'colorscale': 'Viridis'})
            ],
            'layout':
                {
#                    'title':'Comparison of different user types',
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'title':'Score (RandomForestRegressor)', 'color':colors['text']},
                    'yaxis':{'color':colors['text'], 'automargin': True},
                }
            },
        style={'marginLeft': marginVal}

        ),

    html.Div(children=[dcc.Markdown(children=test6)],
        style={'textAlign': 'center','color': colors['titles'], 'marginLeft': marginVal, 'marginRight': marginVal,'font-size': 24,'font-weight': 'bold'}
            ),

    dcc.Graph(
        id='graph9',
        figure={
            'data': [
                {'x': q_numbers,
                'y': scenario_A,
                'name': 'A',
                'mode': 'lines',
                'marker': {'size': 12}},
                {'x': q_numbers,
                'y': scenario_B,
                'name': 'B',
                'mode': 'lines',
                'marker': {'size': 12}},
                {'x': q_numbers,
                'y': scenario_C,
                'name': 'C',
                'mode': 'lines',
                'marker': {'size': 12}},
                {'x': q_numbers,
                'y': scenario_D,
                'name': 'D',
                'mode': 'lines',
                'marker': {'size': 12}},
            ],
            'layout':
                {
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)',
                    'xaxis':{'title':'Question number', 'color':colors['text'],'showgrid':False, 'tickvals':q_numbers},
                    'yaxis':{'title':'Question difficulty','color':colors['text'],'showgrid':False},
                }
            },
        style={'marginLeft': marginVal}

        ),

    html.Div(children=[
        html.H4(children='Results', 
                style={'color': colors['text']}),
        html.Table(
        # Header
        [html.Tr([html.Th(col, style={'color': colors['text']}) for col in ['','A','B','C','D']])] +

        # Body
        [html.Tr([html.Td(optim_results[ii][jj], style={'color': colors['text']}) for jj in range(5)]) for ii in range(2)]
            )            
        ],
        style={'marginLeft': marginVal,'padding': 100}
    ),

],
style={'backgroundColor': '#351c75'}
)


if __name__ == '__main__':
    app.run_server(debug=True)


