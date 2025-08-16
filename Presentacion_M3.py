import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Tema Principal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import streamlit as st


random_seed = 333  # Semilla para reproducibilidad de resultados
np.random.seed(random_seed)  # Para reproducibilidad

# Configuración de opciones de visualización para pandas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.max_rows', 15)  # Ajusta el número de filas a mostrar

# Configuraciones extras
sns.set_style('dark')
dark_template = pio.templates['plotly_dark'].to_plotly_json()
dark_template['layout']['paper_bgcolor'] = 'rgba(30, 30, 30, 0.5)'
dark_template['layout']['plot_bgcolor'] = 'rgba(30, 30, 30, 0)'
pio.templates['plotly_dark_semi_transparent'] = go.layout.Template(dark_template)
pio.templates.default = 'plotly_dark_semi_transparent'
set_config(transform_output="pandas")
set_config(display='diagram')
warnings.filterwarnings("ignore")
#%matplotlib inline


#lectura de datos
df=pd.read_csv('winequality-red.csv',encoding='latin-1')
st.sidebar.subheader('Ramos Palacios Juan Alexis')
ejercicio=st.sidebar.ratio(['Introducción','Análisis de Agrupamiento',
                                        'Selección del Modelo','Perfilamiento','Conclusiones'])
