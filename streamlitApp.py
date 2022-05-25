from operator import index
import streamlit as st
from pandas.core.frame import DataFrame
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg
import time
from random import shuffle
import sys
import nltk 
from nltk.corpus import wordnet 
import gc
from collections import defaultdict
import random
import json
import os
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from gsheetsdb import connect
import plotly
import colorlover as cl
import plotly.offline as py

def display_pca_scatterplot_2D(model):

    data = []
    for i in range(-1,model['cluster'].max()+1):
        
        word_vectors = model.loc[model['cluster']==i]
        scat_text = word_vectors['Unnamed: 0']
        two_dim = word_vectors[['two_dim1','two_dim2']].to_numpy()
        trace = go.Scatter(
            x = two_dim[:,0], 
            y = two_dim[:,1],  
            text = scat_text[:],
            name = 'Cluster'+str(i+2),
            textposition = "top center",
            textfont_size = 10,#20
            mode = 'markers+text',
            marker = {
                'size': 10,#10
                'opacity': 0.8,
                'color': i
            }
        )      
        data.append(trace)

            # Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,#1200
        height = 500 #700
        )

    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

def display_pca_scatterplot_3D(model):

    data = []
    for i in range(-1,model['cluster'].max()+1):
        
        word_vectors = model.loc[model['cluster']==i]
        scat_text = word_vectors['Unnamed: 0']
        three_dim = word_vectors[['three_dim1','three_dim2','three_dim3']].to_numpy()
        trace = go.Scatter3d(
            x = three_dim[:,0], 
            y = three_dim[:,1],
            z = three_dim[:,2],
            text = scat_text[:],
            name = 'Cluster'+str(i+2),
            textposition = "top center",
            textfont_size = 10,#20
            mode = 'markers+text',
            marker = {
                'size': 10,#10
                'opacity': 0.8,
                'color': i
            }
        )      
        data.append(trace)

    # Configure the layout
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 700
        )

    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

Antonym_list = ['commitment rejection', 'manager worker', 'feminine masculine', 'globally locally',
                'family work', 'stakeholders spectators', 'discrimination impartial', 'challenge obscurity',
                'seasonal temporary',
                'alliance proprietorship', 'public private', 'details outlines', 'responsible neglect',
                'marketing secret', 'trust mistrust', 'independent dependent', 'integrity corruption',
                'bankruptcy prosperity', 'insiders outsiders', 'teamwork individualism', 'foreigners natives',
                'criminal rightful', 'strategic impulsive', 'environment pollution', 'diversity uniformity',
                'progressive conservative', 'salary goodies', 'innovator follower', 'pressure relax',
                'secure risky', 'remote physical', 'sustainable unsustainable', 'product service',
                'essential luxury', 'digital analogue', 'effortless demanding', 'nurture neglect',
                'professional amateur', 'ambiguity clarity', 'credible deceptive', 'widespread local',
                'freedom captive', 'order disorder',
                'goal task', 'cost revenue', 'demand supply', 'opportunity threat', 'flexible rigid',
                'isolating social', 'international local', 'innovative traditional', 'satisfied unsatisfied',
                'solution problem', 'store online', 'loss profit', 'ethical unethical',
                'beneficial harmful', 'economic overpriced', 'outdated modern', 'transparency obscurity',
                'lease sell', 'technical natural', 'consistent inconsistent', 'growth decline',
                'tangible intangible', 'employees consultant', 'financial artisanal', 'child childless',
                'connected disconnected', 'corporate individual']

st.title("Word Embeddings for Business Entities")
# option = st.sidebar.selectbox('Create your own polar pairs?',('Yes',  'No'))

check = st.sidebar.selectbox('Check for', ('Bias', 'Hofstede', 'PCA'))
 
if (check == 'Bias'):
    company_or_country = st.sidebar.selectbox('Check for', ('Companies', 'Countries'))
    if (company_or_country == 'Countries'):
        antonym_pair = st.sidebar.selectbox("Select the Antonymn pair", Antonym_list)

        antonym_pair = str(antonym_pair.replace(" ", "_"))

        gnews_url = "https://docs.google.com/spreadsheets/d/1wtUfzJOPIuwVPRuOwZ1zU-JFQF5MQ2a_CnQDKGfPOyo/edit?usp=sharing"
        wiki_url = "https://docs.google.com/spreadsheets/d/12mfEh9o9fyop-ChZ1fUt7_5uDK2DNQHkQhsgvpHkiao/edit?usp=sharing"
        twitter_url = "https://docs.google.com/spreadsheets/d/1_0G95RXRVpu1sjrlGsL74yf1pEqStqv9P0DpbWxrCqo/edit?usp=sharing"
        reddit_url = "https://docs.google.com/spreadsheets/d/1c2o_9RhF1j-WO2rXItt8748DXHbM7YRXIuuSRln5scc/edit?usp=sharing"

        conn = connect()
        gnews_rows = conn.execute(f'SELECT * FROM "{gnews_url}"')
        wiki_rows = conn.execute(f'SELECT * FROM "{wiki_url}"')
        twitter_rows = conn.execute(f'SELECT * FROM "{twitter_url}"')
        reddit_rows = conn.execute(f'SELECT * FROM "{reddit_url}"')

        gnews = pd.DataFrame(gnews_rows)
        country = list(gnews['Country'].str.split('_').str[0])
        gnews.set_index('Country', inplace=True)

        wiki = pd.DataFrame(wiki_rows)
        wiki.set_index('Country', inplace=True)

        twitter = pd.DataFrame(twitter_rows)
        twitter.set_index('Country', inplace=True)

        reddit = pd.DataFrame(reddit_rows)
        reddit.set_index('Country', inplace=True)

        country = st.sidebar.multiselect('Select Upto 5 countries', country)

        country_gnews = [i + "_gnews" for i in country]
        country_gnews = gnews.loc[country_gnews]

        country_wiki = [i + "_wiki" for i in country]
        country_wiki = wiki.loc[country_wiki]

        country_reddit = [i + "_reddit" for i in country]
        country_reddit = reddit.loc[country_reddit]

        country_twitter = [i + "_twitter" for i in country]
        country_twitter = twitter.loc[country_twitter]

        trace0 = go.Scatter(
            {
                'x': country_reddit[antonym_pair],
                'y': country,
                'legendgroup': 'Reddit',
                'name': 'Reddit',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][0],
                    'size': 40,
                }
                # 'text': reddit['Country']
            })

        trace1 = go.Scatter(
            {
                'x': country_wiki[antonym_pair],
                'y': country,
                'legendgroup': 'Wikipedia',
                'name': 'Wikipedia',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][2],
                    'size': 40
                }
                # 'text': wiki['Country']
            })

        trace2 = go.Scatter(
            {
                'x': country_twitter[antonym_pair],
                'y': country,
                'legendgroup': 'Twitter',
                'name': 'Twitter',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][8],
                    'size': 40
                }
                # 'text': twitter['Country']
            })

        trace3 = go.Scatter(
            {
                'x': country_gnews[antonym_pair],
                'y': country,
                'legendgroup': 'Google News',
                'name': 'Google News',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][6],
                    'size': 40
                }
                # 'text': gnews['Country']
            })

        layout = go.Layout(
            title='Business Entities',
            hovermode='closest',
            xaxis=dict(
                title=antonym_pair
            ),
            yaxis=dict(
                title='Companies'
            ),
            showlegend=True,
            # CENTER = 0
        )

        fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
        st.plotly_chart(fig)

    elif (company_or_country == 'Companies'):

        antonym_pair = st.sidebar.selectbox("Select the Antonymn pair", Antonym_list)

        antonym_pair = str(antonym_pair.replace(" ", "_"))

        gnews_url = "https://docs.google.com/spreadsheets/d/1tntSqC-U1e8wL0Tt1k5bw-uc5G717N6Et9QtXJEdMe0/edit?usp=sharing"
        wiki_url = "https://docs.google.com/spreadsheets/d/1fNwrr1dzwot1tiRL7uM34U3qg-epqk1BIyWY4m_Yx9E/edit?usp=sharing"
        twitter_url = "https://docs.google.com/spreadsheets/d/1AMuAYCvOM5bmqMmvP6dCEr9xVqdsqk_MojjG4cJSAjo/edit?usp=sharing"
        reddit_url = "https://docs.google.com/spreadsheets/d/1a5a2yuQ4B_Lq-oO6g6ex0m0gf4UsL4j8_A-Zg3d2uxg/edit?usp=sharing"

        conn = connect()
        gnews_rows = conn.execute(f'SELECT * FROM "{gnews_url}"')
        wiki_rows = conn.execute(f'SELECT * FROM "{wiki_url}"')
        twitter_rows = conn.execute(f'SELECT * FROM "{twitter_url}"')
        reddit_rows = conn.execute(f'SELECT * FROM "{reddit_url}"')

        gnews = pd.DataFrame(gnews_rows)
        company = list(set(gnews['Company_Name']))
        gnews.set_index('Company_Name', inplace=True)

        wiki = pd.DataFrame(wiki_rows)
        wiki.set_index('Company_Name', inplace=True)

        twitter = pd.DataFrame(twitter_rows)
        twitter.set_index('Company_Name', inplace=True)

        reddit = pd.DataFrame(reddit_rows)
        reddit.set_index('Company_Name', inplace=True)

        company = st.sidebar.multiselect('Select Upto 5 companies', company)

        # country_gnews = [i+"_gnews" for i in country]
        company_gnews = gnews.loc[company]

        # country_wiki = [i+"_wiki" for i in country]
        company_wiki = wiki.loc[company]

        # country_reddit = [i+"_reddit" for i in country]
        company_reddit = reddit.loc[company]

        # country_twitter = [i+"_twitter" for i in country]
        company_twitter = twitter.loc[company]

        # reddit = reddit.head(5)
        # wiki = wiki.head(5)
        # twitter = twitter.head(5)
        # gnews = gnews.head(5)

        trace0 = go.Scatter(
            {
                'x': company_reddit[antonym_pair],
                'y': company,
                'legendgroup': 'Reddit',
                'name': 'Reddit',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][0],
                    'size': 40,
                },
                # 'text': reddit['Country']
            })

        trace1 = go.Scatter(
            {
                'x': company_wiki[antonym_pair],
                'y': company,
                'legendgroup': 'Wikipedia',
                'name': 'Wikipedia',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][2],
                    'size': 40
                },
                # 'text': wiki['Country']
            })

        trace2 = go.Scatter(
            {
                'x': company_twitter[antonym_pair],
                'y': company,
                'legendgroup': 'Twitter',
                'name': 'Twitter',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][8],
                    'size': 40
                },
                # 'text': twitter['Country']
            })

        trace3 = go.Scatter(
            {
                'x': company_gnews[antonym_pair],
                'y': company,
                'legendgroup': 'Google News',
                'name': 'Google News',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][6],
                    'size': 40
                },
                # 'text': gnews['Country']
            })

        layout = go.Layout(
            title='Business Entities',
            hovermode='closest',
            xaxis=dict(
                title=antonym_pair
            ),
            yaxis=dict(
                title='Companies'
            ),
            showlegend=True,
            # CENTER = 0
        )

        fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
        st.plotly_chart(fig)

if (check == 'Hofstede'):
    pass
    
if(check == 'PCA'):
    pre_trained = st.sidebar.selectbox("Select a pre-trained model", ('Wikipedia','Google News', 'Reddit', 'Twitter' ))

    if(pre_trained == 'Wikipedia'):
        antonyms = st.sidebar.selectbox("Select a antonym set", ('Business Antonyms','Original Polar Antonyms' ))
        if(antonyms == 'Business Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1svpPZ2zw7G_iQsSTuogRAFgReO8X_CrHp7fL1uitLBo/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1xUPWmpttnUx5Mf_k7y4zZxBTKKkXcEAN-lfydDPgoUM/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1XpFjXCUbkcqYDza1-cUtZvcli3NhdWGyiHMd6S8rUMw/edit?usp=sharing"
        if(antonyms == 'Original Polar Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1pgs3qW8k47xFVKN4TEjXlpfksTJaERYCNKaQFxekUcM/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1T4LMmiC-QmDEBhF1uy8OmtlrBYQ23aRTykmhbG3Y5cE/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1vzvJqSRpnZtdjDOi1DAOCOvkBAsqokj83ZnSWBvbVLU/edit?usp=sharing"  

    elif(pre_trained == 'Google News'):
        antonyms = st.sidebar.selectbox("Select a antonym set", ('Business Antonyms','Original Polar Antonyms' ))
        if(antonyms == 'Business Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1yKz9rcSH31DnNG989S_klSDcAqcTxq4i1Jhk73wl4ug/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1zp6Gd8vasd82brthouaNUPvd0ZV2Df5sIepwLXtORck/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1_FvoyeT7nO70jukT5HUe097w29NyQoCISIfpRDv5WRs/edit?usp=sharing"
        if(antonyms == 'Original Polar Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1ff_GKJnBDBb4XaMx64Y5TqYxWVqVW_SBlQsFRBqO9Yc/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1tulznmXgifzNVubJ-UU3GK2D5pwREbqBYm9WIAnLQz8/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/10PgRzLfvhpWg3XH7YjTcazXv_o1OJfBtP7bnIZ1dfEY/edit?usp=sharing"

    elif(pre_trained == 'Reddit'):
        antonyms = st.sidebar.selectbox("Select a antonym set", ('Business Antonyms','Original Polar Antonyms' ))
        if(antonyms == 'Business Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1-MvGo9EgEDTc9joPBLPtiAnMSTQHYm1MrETD5zp7Tfo/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1ElHqYvfqbGXDEBLg1TUjyvW7ozSuzheY-x4fPq5xu5o/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1v51TTUPz8uAbWni9U9tWTkQwFGXfV1p8WWLBaXq6hyo/edit?usp=sharing"
        if(antonyms == 'Original Polar Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1H1pvjf5ppMsK8RGSPsXqct7-Q49GarWmZ6FayYaCln4/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/14PZK-GOppWLNuEXXdBI_JCtldZfpyYErSC7IDu532V8/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1GSU-SXjWgiaN9sBg6PJUMcLli8xCjoo28a81peqzG-Y/edit?usp=sharing" 

    elif(pre_trained == 'Twitter'):
        antonyms = st.sidebar.selectbox("Select a antonym set", ('Business Antonyms','Original Polar Antonyms' ))
        if(antonyms == 'Business Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/13dh-3rnJ1HiU5V1Juoc2GDxXiFC6iWdaSWPsAieSVAM/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1i8GsCcxV0tbe-T9GdmB-ML9lpZG5i9BsLKVdozkhzjc/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1t9McYMoUFB4BVsri73rBAuzx7xPqSNWxpWkTpuEPKyE/edit?usp=sharing"
        if(antonyms == 'Original Polar Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1b-YHAzrDp63g2PQ3sLMGbYER3_ir0rPgluz6Sww3b-k/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1St8SPiAPq4InWKY3Yp1tXSNghPraY_fnSzQEU5Nomkc/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1LJ_CYDiMifY397X-FtSnTYvbBqyg3CZniOLj2dvAjEg/edit?usp=sharing"        

    conn = connect()

    common = conn.execute(f'SELECT * FROM "{common}"')
    company_url = conn.execute(f'SELECT * FROM "{company_url}"')
    company_url = pd.DataFrame(company_url)
    common = pd.DataFrame(common)

    new_df=pd.concat([common, company_url])  
    df_cluster=new_df.loc[:,new_df.columns!='name']          
    dimension = st.sidebar.selectbox("Select Dimensions", ('2D','3D' ))

    if(dimension == '2D'):
        eps = st.sidebar.slider('Select epsilon', 0.0, 1.0, 0.4)
        min_samples = st.sidebar.slider('Select minimum samples', 1, 10, 5)
        dbscan = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)#high eps low samples only clusters common, 
        cluster_labels = dbscan.fit_predict(df_cluster)

        two_dim = PCA(random_state=0).fit_transform(df_cluster)[:,:2]
        df_cluster[['two_dim1','two_dim2']]=two_dim.tolist()
        df_cluster['cluster']=cluster_labels
        df_cluster['Unnamed: 0']=new_df['name']

        display_pca_scatterplot_2D(df_cluster[:])

    elif(dimension == '3D'):

        eps = st.sidebar.slider('Select epsilon', 0.0, 1.0, 0.4)
        min_samples = st.sidebar.slider('Select minimum samples', 1, 10, 5)
        dbscan = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)#high eps low samples only clusters common, 
        cluster_labels = dbscan.fit_predict(df_cluster)
        
        three_dim = PCA(random_state=0).fit_transform(df_cluster)[:,:3]
        df_cluster[['three_dim1','three_dim2','three_dim3']]=three_dim.tolist()
        df_cluster['cluster']=cluster_labels
        df_cluster['Unnamed: 0']=new_df['name']

        display_pca_scatterplot_3D(df_cluster[:])

