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


st.title("Word Embeddings for Business Entities")
option = st.sidebar.selectbox('Create your own polar pairs?',('Yes',  'No'))

if(option == 'Yes'):
    pass

elif(option == 'No'):
    wiki_url = "https://docs.google.com/spreadsheets/d/1bkhvGSLMIKFHfbjdzH6LRunqYpt5x8-tUQvdutUCvaU/edit?usp=sharing"
    conn = connect()
    rows = conn.execute(f'SELECT * FROM "{wiki_url}"')
    df_gsheet = pd.DataFrame(rows)
    st.write(df_gsheet)
# st.write('You selected:', option)
