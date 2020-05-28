
#----- DATA IMPORT AND CLEANING
import numpy as np
import pandas as pd

REMOTE_DATAFILE = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"
#REMOTE_DATAFILE = "COVID-19-geographic-disbtribution-worldwide_200511.xlsx"

def read_coviddata(filename=REMOTE_DATAFILE,popThr=200000):
    "reads and cleans covid data. Countries with population below popThr are excluded"

    dfw = pd.read_excel(filename,parse_dates = ['dateRep'])
    # delete weird cases
    dfw.drop(dfw[dfw.countriesAndTerritories=='Cases_on_an_international_conveyance_Japan'].index,inplace=True)
    # sort by country and date
    dfw = dfw.sort_values(by=['countriesAndTerritories','dateRep'])
    # reset ix
    dfw=dfw.reset_index(drop=True)

    # col renaming
    dfw = dfw.rename(columns={'dateRep':'date',
                              'countriesAndTerritories':'country',
                              'countryterritoryCode':'countryCode',
                              'popData2018':'population',
                              'continentExp':'continent'})
    # drop cols
    dfw= dfw.drop(columns=['day','month','year'])
    # remove countries under pop threshold
    dfw = dfw[dfw.population>popThr]
    # remove leading zeros by country
    dfw=dfw.groupby('country',as_index=False).apply(remove_leading_zeros)
    # reset ix
    dfw=dfw.reset_index(drop=True)
    # add daycount
    dfw['daycount'] = (dfw['date']-dfw['date'].min()).apply(lambda x: x.days)
    # add trends and relative delta
    dfw['cases_trend'] = dfw.groupby('country')['cases'].rolling(center=True,window=7).mean().values
    dfw['cases_relative_delta'] = dfw['cases']/dfw['cases_trend']-1
    dfw['deaths_trend'] = dfw.groupby('country')['deaths'].rolling(center=True,window=7).mean().values
    dfw['deaths_relative_delta'] = dfw['deaths']/dfw['deaths_trend']-1
    #
    return dfw

def remove_leading_zeros(sub):
    "removes the initial date points where cases and deaths are zero"
    
    zeroFlag = (sub.cases==0) & (sub.deaths==0)
    if not zeroFlag.iloc[0]:
        return sub
    #indices where flag is true
    ix0 = zeroFlag[zeroFlag].index
    # diff of indices
    delta_ix0 = np.diff(ix0)
    # jump indices
    jump_ix=np.where(delta_ix0>1)
    # if no jump, there is only one section with zeros
    if list(jump_ix[0])==[]:
        first_discontinuity_ix= -1
    else:
        first_discontinuity_ix = jump_ix[0][0]
    #
    out = sub.loc[ix0[first_discontinuity_ix]+1:]
    #
    return out

#---- MATH FUNCTIONS
def lnfit(x,y):
    "Fits linearly the log(y) - natural log"
    # remove points where log does not exist
    x=x[y>0]
    y=y[y>0]
    f = np.polyfit(x, np.log(y),1)
    return f

def lnval(f, inp):
    "Evaluate exp**fit(x)"
    return np.exp(np.polyval(f,inp))

import warnings
def relative_growth(daily_series):
    "Calculates the relative growth starting from a daily series"
    cum_series = daily_series.cumsum()
    # suppress warnings as you might have division by zero
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        start = np.array([np.nan])
        rv = daily_series.values[1:]/cum_series.values[:-1] # relative variation
        alldata = np.concatenate([start,rv])
    #
    return pd.Series(alldata,index=daily_series.index)


def adaptive_moving_average(y, window=7):
    """centered moving average with window reduction at the extremities.
    Window size must be uneven"""
    assert window%2 == 1
    # standard moving average
    yout = y.rolling(window=window,center=True).mean()
    # extremities
    nextremity = int((window-1)/2)
    for k in range(nextremity):
        yout.iloc[k] = y.iloc[0:2*k+1].mean()
        yout.iloc[-(k+1)] = y.iloc[-(2*k+1):].mean()
    #
    return yout

#---- STATISTICS CALCULATIONS
def clean_std(df,field_to_proc,trendThr=10,minNPoints=5,consecFlag=False):
    """calculates the standard deviation of the portion of signal where trend is above a threshold.
    A minimum number of points is required to calculate the std (minNPoints)
    The consecFlag dictates whether the portion of signal above trendThr has to be a single consecutive block
    or can result in multiple segments
    The indices above the threshold have to be consecutive otherwise nan is reported.
    Field_to_proc is either cases or deaths"""
    #
    assert (field_to_proc=='cases') or (field_to_proc=='deaths')
    # remove below thr
    df= df[ df[field_to_proc+'_trend']>trendThr ] 
    # consecutive check if consecFlag is true
    if consecFlag:
        consecCheck = all(np.diff(df.index)==1)
    else:
        consecCheck = True
    #
    if len(df)>minNPoints and consecCheck:
        return df[field_to_proc+'_relative_delta'].std()
    else:
        return np.nan

def data_variability(df,casesThr=50,deathsThr=10):
    "produces clean_std for each country on cases and deaths signals"
    cases_sigma = df.groupby('country').apply(lambda x: clean_std(x,'cases',casesThr)).to_frame(name='cases_rel_sigma')
    deaths_sigma = df.groupby('country').apply(lambda x: clean_std(x,'deaths',deathsThr)).to_frame(name='deaths_rel_sigma')
    sigmas = pd.concat((cases_sigma,deaths_sigma),axis=1)
    return sigmas


#---- SIGNAL EXTRACTION

from datetime import timedelta

def extract_signal(dfw,sigmas,country,yvalue,signaltype,popFlag,maFlag,extrapFlag,
                   extrap_last_ndays=15,extrap_ndays=60,refPop=1e4):
    # signal type
    assert signaltype in ['daily','relative growth','cumulative']
    #
    sub = dfw[dfw.country==country] 
    x = sub['date']
    
    daily = sub[yvalue]
    cumulative = np.cumsum(daily)
    relgrowth = relative_growth(daily)
    #
    extra_daily = pd.Series([],dtype='float64')
    extra_cumulative = pd.Series([],dtype='float64')
    extra_growth = pd.Series([],dtype='float64')
    newdays = pd.Series([],dtype='float64')
    pred_ints = pd.Series([],dtype='float64')
    # EXTRAPOLATION
    if extrapFlag==['Extrap']:
        # calculate relative growth from daily series
        # extract last data points
        finaldaycounts = sub['daycount'].iloc[-extrap_last_ndays:]
        finalgrowths = relgrowth.iloc[-extrap_last_ndays:]
        # try ln interpolation
        try:
            resfit = lnfit(finaldaycounts,finalgrowths)
        except:
            resfit = []
        #
        # extrapolate only if fit exists and has negative slope
        if type(resfit)==np.ndarray:
            lastdaycount = sub['daycount'].iloc[-1]
            lastcollecteddate = sub['date'].iloc[-1]
            # extra days
            extra_daycount = np.arange(lastdaycount+1,lastdaycount+extrap_ndays+1)
            newdays = pd.Series(pd.date_range(lastcollecteddate+timedelta(days=1)
                                    ,lastcollecteddate+timedelta(days=extrap_ndays)) )
            #
            if resfit[0]<0:
                # total values at last collection date
                #total = sub[yvalue].sum()
                total = cumulative.iloc[-1]
                # extrapolate with negative slope
                extra_growth = lnval(resfit,extra_daycount)
                extra_cumulative = total*np.cumprod(1+extra_growth)
                extra_daily = np.concatenate( (np.array([extra_cumulative[0]-total]),np.diff(extra_cumulative)) ) # array
                #

                # IF EXTRAP IS SUCCESSFUL, CALC PREDICTION INTERVALS
                sigma = sigmas.loc[country,yvalue+'_rel_sigma']
                if np.isnan(sigma) or signaltype=='relative growth':
                    pred_ints = pd.Series([],dtype='float64')
                else:
                    if signaltype=='daily':
                        upper = extra_daily*(1+1.96*sigma)
                        lower = extra_daily*(1-1.96*sigma)
                        # check <0
                        lower[lower<0]=0
                        #
                    elif signaltype=='cumulative':
                        sigmacum = np.sqrt(np.cumsum(extra_daily**2))*sigma
                        upper = extra_cumulative +1.96*sigmacum
                        lower = extra_cumulative -1.96*sigmacum
                        # check <0
                        lower[lower<0]=0
    
                    pred_ints = pd.DataFrame({'lower':lower,'upper':upper},index=newdays)


            else:
                # with positive slope just report nan's
                tmp = np.zeros((len(newdays),))
                tmp[:]= np.nan
                extra_daily = tmp
                extra_cumulative = tmp
                extra_growth = tmp
                pred_ints = pd.Series([],dtype='float64')
                
                                            
    # concatenate
    x= pd.concat((x,newdays),axis=0)

    if signaltype == 'daily':
        y = daily
        yextra = extra_daily
    elif signaltype =='cumulative':
        y = cumulative
        yextra = extra_cumulative
    elif signaltype == 'relative growth':
        y = relgrowth
        yextra = extra_growth

    # Moving average
    if maFlag == ['MA']:
        y = adaptive_moving_average(y)
    #
    y= pd.concat((y,pd.Series(yextra)),axis=0)

        
    # pop norm
    if popFlag==['Norm'] and signaltype!='relative growth': # do not scale by pop if relgrowth
        y = y/sub['population'].iloc[0]*refPop
        pred_ints = pred_ints/sub['population'].iloc[0]*refPop

    time_series = pd.Series(data=y.values,index=x.values)

    return time_series, pred_ints

#----- APP SETUP
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import timedelta
import plotly
# extract color list
color_list = plotly.colors.qualitative.Plotly

# read data file
dfw = read_coviddata()
# countries
countries = dfw.country.unique()
# calculate relative std's
sigmas = data_variability(dfw)

#
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)


# app layout
app.title='My Covid19 Dashboard'
app.layout = html.Div(children=[
    html.H1('Covid19 World Status',style={'textAlign':'center'}),

    html.Div(children=[
        html.Label('Countries'),
        dcc.Dropdown(id='country_selection',options=[{'label': x,'value':x} for x in countries], 
                    value=['Italy'],multi=True),
        #
        html.Label('Y-values'),
        dcc.Dropdown(id='ycolumn',options=[ {'label': 'Cases','value':'cases'}, 
                                            {'label': 'Deaths','value':'deaths'} ],value='cases'),
        #
        html.Div(children=[
        html.H6('Signal type'),
        dcc.RadioItems(
                id='signal-type',
                options=[{'label': i, 'value': i.lower()} for i in ['Daily', 'Cumulative','Relative Growth']],
                value='daily',
                ),]),
        
        html.H6('Y-scale'),
        dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i.lower()} for i in ['Linear', 'Log']],
                value='linear',
                labelStyle={'display': 'inline-block'}),
        html.H6('Options'),
        #
        dcc.Checklist(  id='norm-flag',
                        options=[{'label':'Population Normalise','value':'Norm'}],value=[]),
        dcc.Checklist(  id='ma-flag',
                        options=[{'label':'Moving Average','value':'MA'}],value=[]),
        html.H6('Time extrapolation'),
        dcc.Checklist(  id='extrap-flag',
                        options=[{'label':'Show Projection','value':'Extrap'}],value=[]),
        dcc.Checklist(  id='predints-flag',
                        options=[{'label':'Prediction Intervals','value':'PredInts'}],value=['PredInts']),
        html.Label('R. Bianco 2020')

        ]
        ,style={'width': '30%', 'display': 'inline-block'}
        ),

    dcc.Graph(id='timeplot',style={'width':'70%','float':'right'})
])

# APP callback
@app.callback(
    Output('timeplot', 'figure'),
    [Input('country_selection', 'value'),
     Input('ycolumn', 'value'),
     Input('signal-type','value'),
     Input('yaxis-type', 'value'),
     Input('norm-flag','value'),
     Input('ma-flag','value'),
     Input('extrap-flag','value'),
     Input('predints-flag','value')])
def update_graph(countries,yvalue,signaltype,yaxistype,popFlag,maFlag,extrapFlag,predintsFlag):
    
    # traces
    traces = []
    for ix, country in enumerate(countries):
        
        time_series, pred_ints = extract_signal(dfw,sigmas,country,yvalue,signaltype,popFlag,maFlag,extrapFlag)
        #
        clean_ix = ix%len(color_list)
        traces.append( dict(x=time_series.index, y=time_series.values,mode='lines',name=country,
                            line={'color':color_list[clean_ix]}))
        # pred intervals
        if predintsFlag==['PredInts'] and len(pred_ints.values)>0:            
            #upper
            traces.append(dict(x=pred_ints.index, y=pred_ints.upper,mode='lines',
                            line={'color':color_list[clean_ix],'width':0},showlegend=False))
            #lower
            traces.append(dict(x=pred_ints.index, y=pred_ints.lower,mode='lines',
                            line={'color':color_list[clean_ix],'width':0},fill='tonexty',alpha=.3,showlegend=False))
    # layout
    if popFlag == ['Norm'] and signaltype!='relative growth':
        normLab = ' [per 10.000 people]'
    else:
        normLab = ''
    #
    if maFlag ==['MA']:
        maLab= ' (Moving Average)'
    else:
        maLab= ''
    #
    ytitle = signaltype.capitalize() + normLab + maLab
    # 
    layout = dict(
                    xaxis={'title':'date'},
                    yaxis={'type':yaxistype,'title':ytitle},
                    margin={'l': 80, 'b': 40, 't': 50, 'r': 10},
                    title= {'text':yvalue.capitalize(),'font':{'size':25}},
                    )
    # Add rectangle and annotation if Extrap is required
    if extrapFlag == ['Extrap'] and len(countries)>0:
        lastcollecteddate = dfw['date'].max()
        lastplotdate = time_series.index.max()
        layout['shapes'] =  [  dict(
                                type="rect",
                                # x-reference is assigned to the x-values
                                xref="x",
                                # y-reference is assigned to the plot paper [0,1]
                                yref="paper",
                                x0=lastcollecteddate+timedelta(days=1),
                                y0=0,
                                x1=lastplotdate,
                                y1=1.05,
                                fillcolor="LightGreen",
                                opacity=0.1,
                                layer="below",
                                line={'width':0},)
        ]
        layout['annotations'] = [  dict(
                                        text=' Projection',
                                        xanchor='left',
                                        showarrow=False,
                                        # x-reference is assigned to the x-values
                                        xref="x",
                                        # y-reference is assigned to the plot paper [0,1]
                                        yref="paper",
                                        x=lastcollecteddate,
                                        y=1.05,
                                        font= {'size':18})
        
        ]
    # 
    return {'data':traces,
            'layout':layout}



#import os
#os.startfile("http://127.0.0.1:8050/")

if __name__ == '__main__':
    app.run_server(debug=True)

