from operator import index
import streamlit as st
from pandas.core.frame import DataFrame
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm_notebook as tqdm
from random import shuffle
from nltk.corpus import wordnet
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from gsheetsdb import connect
import plotly
import colorlover as cl
import plotly.offline as py
import plotly.graph_objs as go
            
Antonym_list = ['commitment rejection', 'manager worker', 'feminine masculine', 'globally locally',
 'family work', 'stakeholders spectators', 'discrimination impartial', 'challenge obscurity', 'seasonal temporary',
 'alliance proprietorship', 'public private', 'details outlines', 'responsible neglect',
 'marketing secret', 'trust mistrust', 'independent dependent', 'integrity corruption',
 'bankruptcy prosperity', 'insiders outsiders', 'teamwork individualism', 'foreigners natives',
 'criminal rightful', 'strategic impulsive', 'environment pollution', 'diversity uniformity',
 'progressive conservative', 'salary goodies', 'innovator follower', 'pressure relax',
 'secure risky', 'remote physical', 'sustainable unsustainable','product service',
 'essential luxury', 'digital analogue', 'effortless demanding','nurture neglect',
 'professional amateur', 'ambiguity clarity', 'credible deceptive', 'widespread local', 'freedom captive', 'order disorder',
 'goal task', 'cost revenue', 'demand supply', 'opportunity threat', 'flexible rigid',
 'isolating social', 'international local', 'innovative traditional', 'satisfied unsatisfied',
 'solution problem', 'store online', 'loss profit', 'ethical unethical',
 'beneficial harmful', 'economic overpriced', 'outdated modern', 'transparency obscurity',
 'lease sell', 'technical natural', 'consistent inconsistent', 'growth decline',
 'tangible intangible', 'employees consultant', 'financial artisanal', 'child childless',
 'connected disconnected', 'corporate individual']

st.title("Word Embeddings for Business Entities")
# option = st.sidebar.selectbox('Create your own polar pairs?',('Yes',  'No'))

check = st.sidebar.selectbox('Check for',('Bias',  'Hofstede'))

if(check == 'Bias'):
    company_or_country = st.sidebar.selectbox('Check for',('Companies',  'Countries'))
    if(company_or_country == 'Countries'):
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
        gnews.set_index('Country', inplace= True)

        wiki = pd.DataFrame(wiki_rows)
        wiki.set_index('Country', inplace= True)

        twitter = pd.DataFrame(twitter_rows)
        twitter.set_index('Country', inplace= True)

        reddit = pd.DataFrame(reddit_rows)
        reddit.set_index('Country', inplace= True)

        # url = 'https://drive.google.com/file/d/1TsqXgNiaJ_uaRnP31ugQCiPF-wo5Gxv7/view?usp=sharing'
        # path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        # df = pd.read_csv(path)
        # st.write(df)


        # st.write(reddit)

        # st.write(country)

        country = st.sidebar.multiselect('Select Upto 5 countries', country)
#         st.write(country)

        country_gnews = [i+"_gnews" for i in country]
        country_gnews = gnews.loc[country_gnews]

        country_wiki = [i+"_wiki" for i in country]
        country_wiki = wiki.loc[country_wiki]

        country_reddit = [i+"_reddit" for i in country]
        country_reddit = reddit.loc[country_reddit]

        country_twitter = [i+"_twitter" for i in country]
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

        trace1= go.Scatter(
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
            title= 'Business Entities',
            hovermode = 'closest',
            xaxis = dict(
                title= antonym_pair
            ),
            yaxis = dict(
                title='Companies'
            ),
            showlegend = True,
            # CENTER = 0
        )

        fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
        st.plotly_chart(fig)

    elif(company_or_country == 'Companies'):

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
        gnews.set_index('Company_Name', inplace= True)

        wiki = pd.DataFrame(wiki_rows)
        wiki.set_index('Company_Name', inplace= True)

        twitter = pd.DataFrame(twitter_rows)
        twitter.set_index('Company_Name', inplace= True)

        reddit = pd.DataFrame(reddit_rows)
        reddit.set_index('Company_Name', inplace= True)


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

        trace1= go.Scatter(
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
            title= 'Business Entities',
            hovermode = 'closest',
            xaxis = dict(
                title=antonym_pair
            ),
            yaxis = dict(
                title='Companies'
            ),
            showlegend = True, 
            # CENTER = 0
        )

        fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
        st.plotly_chart(fig)











    # embeddings = st.sidebar.selectbox('Select pre-trained embbeddings?',('Wikipedia',  'Google News', 'Twitter', 'Reddit'))
    # if(embeddings == 'Wikipedia'):
    #     url = "https://docs.google.com/spreadsheets/d/1bkhvGSLMIKFHfbjdzH6LRunqYpt5x8-tUQvdutUCvaU/edit?usp=sharing"
    # elif(embeddings == 'Google News'):
    #     url = "https://docs.google.com/spreadsheets/d/1AiU_rhYuBphWByYszFXtFHDdGQ8yBXvqBvLEhuOEsgk/edit?usp=sharing"
    # elif(embeddings == 'Twitter'):
    #     url = "https://docs.google.com/spreadsheets/d/1dA6wI4ut8xhcDV0NRVdNq6hPKvgUhbWnroo_z2WFVf8/edit?usp=sharing"
    # elif(embeddings == 'Reddit'):
    #     url = "https://docs.google.com/spreadsheets/d/1uq5ZwkVIYBj7NLb0jtm2bEo-LZmO57TeZ-f4k8DMFsE/edit?usp=sharing"
    # conn = connect()
    # rows = conn.execute(f'SELECT * FROM "{url}"')
    # df_gsheet = pd.DataFrame(rows)
    # st.write(df_gsheet)
# st.write('You selected:', option)
