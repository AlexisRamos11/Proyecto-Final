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

final_columns = df.select_dtypes(include=["number"]).columns



ejercicio=st.sidebar.radio('Proyecto Final',['Introducción','Selección de Variables','Análisis de Agrupamiento',
                                        'Selección del Modelo','Perfilamiento','Conclusiones'])


if ejercicio=='Introducción':

    st.header('Red and White Variants of the Portuguese Wine')
    st.header('Introducción')
    st.markdown('''
        El Vinho Verde es un vino distintivo producido en la región noroeste de Portugal, conocida como Minho. A pesar de su nombre, que en portugués significa "vino verde", no se refiere a su color, sino a su juventud, ya que generalmente se embotella y consume poco tiempo después de su producción. Aunque es más conocido por sus versiones blancas, el Vinho Verde también cuenta con variantes tintas que poseen un carácter único y menos difundido internacionalmente. Estas variantes tintas suelen elaborarse a partir de variedades autóctonas como Vinhão, Borraçal y Amaral, ofreciendo vinos de color intenso, con acidez marcada, taninos firmes y notas frutales profundas.
                

En el caso del Vinho Verde tinto, sus características sensoriales y de calidad están fuertemente influenciadas por factores fisicoquímicos, como la acidez fija, el pH, el contenido de alcohol y los niveles de dióxido de azufre, así como por elementos relacionados con la dulzura y la estructura, como el azúcar residual y los cloruros. Estas propiedades no solo determinan la percepción gustativa del vino, sino que también inciden en su preservación y perfil aromático.
    ''')
    st.subheader('Objetivo')
    st.markdown('''
        A partir del dataset [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) analizar y segmentar vinos tintos de
la denominación Vinho Verde de Portugal a partir de sus propiedades
fisicoquímicas y sensoriales, empleando técnicas de modelación no
supervisada como K-Means y clustering jerárquico y poder identificar la
mejor calidad de vinos.
Para ello, se considerarán las siguientes variables:
- **fixed acidity (Acidez fija):** La mayoría de los ácidos presentes en el
vino son fijos o no volátiles, es decir, no se evaporan fácilmente.
                
- **volatile acidity (Acidez volátil):** Cantidad de ácido acético en el vino.
En niveles altos, puede provocar un sabor desagradable similar al
vinagre.
- **citric acid (Ácido cítrico):** Presente en pequeñas cantidades; puede
aportar frescura y sabor al vino.
- **residual sugar (Azúcar residual):** Cantidad de azúcar que queda
después de que finaliza la fermentación. Es raro encontrar vinos con
menos de 1 g/L, y los que tienen más de 45 g/L se consideran dulces.
- **chlorides (Cloruros):** Cantidad de sal presente en el vino.
- **free sulfur dioxide (Dióxido de azufre libre):** Forma libre de SO₂ que
existe en equilibrio entre SO₂ molecular (gas disuelto) y el ion bisulfito.
Previene el crecimiento microbiano y la oxidación del vino.
- **total sulfur dioxide (Dióxido de azufre total):** Cantidad total de SO₂
libre y combinado. En bajas concentraciones es prácticamente
indetectable, pero cuando el SO₂ libre supera los 50 ppm, se percibe
en el aroma y sabor del vino.
- **density (Densidad):** Muy cercana a la densidad del agua, dependiendo
del porcentaje de alcohol y contenido de azúcar.
- **pH (pH):** Indica qué tan ácido o básico es un vino, en una escala de 0
(muy ácido) a 14 (muy básico). La mayoría de los vinos están entre 3
y 4.
- **sulphates (Sulfatos):** Aditivo del vino que puede contribuir a los
niveles de SO₂, actuando como antimicrobiano y antioxidante.
- **alcohol (Alcohol):** Porcentaje de contenido alcohólico del vino.
- **quality (Calidad sensorial):** Variable de salida basada en datos
sensoriales, con una puntuación entre 0 y 10.

    ''')

elif ejercicio=='Selección de Variables':
    st.header('Selección de Variables')
    st.markdown(f'''
        Podemos observar que para esta situación contamos con una estructura del dataset {df.shape}, con los siguientes datos:

    ''')
    st.dataframe(df)
    st.markdown('Además podemos visualizar que todas nuestras variables son numéricas con las siguiente estadísticas:')
    st.dataframe(df.describe())

    st.markdown('De mismo modo podemos observar que nuestro dataset no cuenta con valores nulos, lo que nos ayuda a no realizar mucha limpieza:')
    st.dataframe((df.isnull().sum()).reset_index().sort_values(0,ascending=False))
elif ejercicio=='Análisis de Agrupamiento':
    st.header('Análisis de Agrupamiento')
    #st.markdown('Matriz de Correlación')
    #st.dataframe(df[final_columns].corr())

    #matriz de correlacion
    graf_1=plt.figure(figsize=(10, 8))
    sns.heatmap(df[final_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlación")
    
    #st.pyplot(graf_1)

    X = df[final_columns]
    #se estandarizan los datos antes de aplicar T-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_seed)
    X_pca = pca.fit_transform(X_scaled)

    st.subheader('Evaluación de clusters (Métricas de segmentación)')
    st.markdown('Para este dataset se implementaron dos modelos de agrupación Kmeans y Cluster-Jerárquico.')
    st.subheader('1. Clustering Jerárquico')
    # Calculamos el linkage para el dendograma
    Z = linkage(X_pca, method='ward')

    fig_CJ=plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
    plt.title("Dendrograma - Clustering Jerárquico")
    plt.xlabel("Cluster Size")
    plt.ylabel("Distancia")
    st.pyplot(fig_CJ)
    st.markdown('Se observa en el anterior gráfico el uso del Dendograma para poder identificar el número de componentes a elegir, en este caso para poder visualizar después los clusters con T-sne.')

    st.markdown('Se elige el numero de clusters para el dendograma, aqui se puede obseervar que son 2, por lo que procederemos a usar cluster jerárquico con t=2.')
    df['Cluster_Jerarquico'] = fcluster(Z, t=2, criterion='maxclust')

    st.markdown('Ahora observemos cupantos valores tenemos por cada cluster.')
    st.dataframe(df['Cluster_Jerarquico'].value_counts())

    st.subheader('T-SNE para Clustering-Jerarquico')
    # Visualización t-SNE
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)

    df_tsne=X_tsne
    df_tsne['Grupos']=df['Cluster_Jerarquico']

    #Creacion de grafico agrupado
    fig_tsne=plt.figure(figsize=(8,6))
    sns.scatterplot(x='tsne0', y='tsne1', data=df_tsne,hue='Grupos')
    plt.title('Gráfico Agrupado T-SNE para Clustering-Jerarquico')
    #plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    st.pyplot(fig_tsne)


    st.subheader('2. K-Means')
    st.markdown('Para el caso de k-means se aplicó el método del codo y silhouette para poder identificar el k-óptimo y así poder definir el número de clusters y de mismo modo se aplicó t-sne para poder visualizar los agrupamientos.')
    k_range = range(3, 11)
    inertias = []
    silhouettes = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        kmeans.fit(X_pca)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_pca, kmeans.labels_))

    # Gráfico del método del codo y silhouette
    fig_kmeans_1=plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, 'go-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Puntaje Silhouette')
    plt.title('Análisis Silhouette')
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(fig_kmeans_1)

    # k óptimo (máximo silhouette)
    optimal_k = k_range[np.argmax(silhouettes)]
    st.markdown(f"\nNúmero óptimo de clusters: {optimal_k} (Silhouette={max(silhouettes):.3f})")

    # Aplicamos K-Means con k óptimo
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, random_state=random_seed)
    clusters = kmeans.fit_predict(X_pca)
    df['Clusters']=clusters

    st.subheader('T-SNE para K-Means')
    # Visualización t-SNE
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)

    df_tsne=X_tsne
    df_tsne['Grupos']=df['Clusters']

    #Creacion de grafico agrupado
    fig_tsne_k=plt.figure(figsize=(8,6))
    sns.scatterplot(x='tsne0', y='tsne1', data=df_tsne,hue='Grupos')
    plt.title('Gráfico Agrupado T-SNE para K-Means')
    #plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    st.pyplot(fig_tsne_k)

elif ejercicio=='Selección del Modelo':
    st.balloons()
    st.header('Selección del Modelo')
    st.markdown('Podemos concluir que nuestro modelo seleccionado es **Cluster-Jerarquico**')
    st.markdown('''
        -	Se puede elegir el número de clusters a partir del Dendograma de manera visual.
-	Captura mejor la estructura real en el dataset debido a la dispersión en la que se encuentra.
-	La cantidad de los vinos (el tamaño del dataset) ayuda a que se ajuste mejor al cluster jerárquico.
-	Ofrece una mejor jerarquía de agrupamientos y captura mejor los datos de los vinos.

                ''')



elif ejercicio =='Perfilamiento':

    st.header('Perfilamiento')
    st.markdown('El siguiente perfilado se realizó a partir de un agrupamiento por cluster y además aplicando un criterio de promedio para poder definir que cluster es mejor.')
    
    #___
    X = df[final_columns]
    #se estandarizan los datos antes de aplicar T-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_seed)
    X_pca = pca.fit_transform(X_scaled)
    Z = linkage(X_pca, method='ward')
    df['Cluster_Jerarquico'] = fcluster(Z, t=2, criterion='maxclust')
    #__
    perfil = df.groupby('Cluster_Jerarquico').mean(numeric_only=True)

    #valores altos y bajos para cada grupo
    for col in perfil.columns:
        max_cluster = perfil[col].idxmax()
        min_cluster = perfil[col].idxmin()
        print(f"{col}: más alto en cluster {max_cluster}, más bajo en cluster {min_cluster}")


    tabla_perfilados = {
    "Variable": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                 "pH", "sulphates", "alcohol", "quality"],
    "Cluster 1 (Alta Calidad)": [9.69, 0.4, 0.45, 2.71, 0.096, 13.24, 36.29, 0.99736, 3.22, 0.74, 10.74, 6.04],
    "Cluster 2 (Rústicos)": [7.43, 0.61, 0.16, 2.43, 0.082, 17.59, 53.07, 0.99635, 3.37, 0.6, 10.22, 5.37],
    "Interpretación": [
        "Mayor frescura y estructura en Cluster 1.",
        "Cluster 2 con notas más avinagradas y agresivas.",
        "Mayor aporte de frescura y aromas frutales en Cluster 1.",
        "Ligero dulzor adicional en Cluster 1, más redondez.",
        "Mayor aporte salino y complejidad en Cluster 1.",
        "Cluster 2 con mayor protección microbiológica.",
        "Cluster 2 más protegido contra oxidación.",
        "Mayor cuerpo en Cluster 1.",
        "Cluster 2 con acidez percibida más suave.",
        "Mayor estabilidad y estructura en Cluster 1.",
        "Cluster 1 con más calidez alcohólica.",
        "Mejor valoración sensorial en Cluster 1."
        ]
        }
    st.dataframe(tabla_perfilados)

    st.markdown('''
        - **Cluster 1:** Se observa que para el cluster 1 podemos encontrar vinos frescos con menor acidez, mayor alcohol. Se encuentra equilibrado y con posible opción a ser el mejor cluster.
        - **Cluster 2:** Se observa que la acidez volátil es mayor con una menor calidad sensorial y menor cuerpo.

                ''')

elif ejercicio == 'Conclusiones':
    st.header('Conclusiones')
    st.markdown('El cluster 1 nos indica un mejor agrupamiento para la caldiad sensorial, mayor cuerpo, frescura y estructura por lo cual podemos definir que la calidad del vino es excelente. Por lo tanto, el agrupamiento por el cluster 1 es el que mejor rendimiento en cuanto calidad nos podrá brindar.')
    
    st.subheader('Cateo de Vino')
    st.markdown('Se adjunta evidencia del cateo del vino del cluster 1 y personalmente confirmo que la calidad es excelente.')
    st.image('Wine.png')
