
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
import dash, dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


path1='CollegeScorecard.csv'

bd = pd.read_csv(path1, sep = ",",  low_memory = False , encoding = 'UTF-8',encoding_errors = 'ignore')

bd1=bd.copy()
bd1.dropna(thresh = bd1.shape[0]*1, how = 'all', axis = 1, inplace = True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

bd2 = bd1.select_dtypes(include=numerics)
semillas_2=bd2.drop(columns=['UNITID',
 'OPEID',
 'opeid6',
 'NUMBRANCH',
 'PREDDEG',
 'st_fips',],)


semillas_2["HCM2"].replace(0, "not under investigation", inplace=True)
semillas_2["HCM2"].replace(1, "under investigation", inplace=True)
semillas_2["main"].replace(1, "Main campus", inplace=True)
semillas_2["main"].replace(0, "branch", inplace=True)
semillas_2["CURROPER"].replace(0, "closed", inplace=True)
semillas_2["CURROPER"].replace(1, "operating", inplace=True)
semillas_2["region"].replace(0, "U.S", inplace=True)
semillas_2["region"].replace(1, "New England", inplace=True)
semillas_2["region"].replace(2, "Mid East", inplace=True)
semillas_2["region"].replace(3, "Great Lakes", inplace=True)
semillas_2["region"].replace(4, "Plains", inplace=True)
semillas_2["region"].replace(5, "Southeast", inplace=True)
semillas_2["region"].replace(6, "Southwest", inplace=True)
semillas_2["region"].replace(7,"Rocky Mountains", inplace=True)
semillas_2["region"].replace(8, "Far West", inplace=True)
semillas_2["region"].replace(9, "Outlying Areas", inplace=True)

semillas_2["CONTROL"].replace(1, "Public", inplace=True)
semillas_2["CONTROL"].replace(2, "Private Non Profit", inplace=True)
semillas_2["CONTROL"].replace(3, "Private for profit", inplace=True)

semillas_2["HIGHDEG"].replace(0, "Non-degree-granting", inplace=True)
semillas_2["HIGHDEG"].replace(1, "Certificate degree", inplace=True)
semillas_2["HIGHDEG"].replace(2, "Associate degree", inplace=True)
semillas_2["HIGHDEG"].replace(3, "Bachelor's degree", inplace=True)
semillas_2["HIGHDEG"].replace(4, "Graduate degree", inplace=True)

data_categorica_dummies=pd.get_dummies(semillas_2)
semillas33=np.array(data_categorica_dummies)

from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
ayuda=cluster.fit(data_categorica_dummies)
labels_2D=ayuda.labels_


model=AgglomerativeClustering(n_clusters=3, linkage='ward')

data_fit_3=model.fit(data_categorica_dummies)
lab_3c=data_fit_3.labels_
data_categorica_dummies['Labels_3Clusters']=lab_3c


is_G=data_categorica_dummies.loc[:, 'Labels_3Clusters']==0
C3_G=data_categorica_dummies[is_G]

df_2 = pd.DataFrame(C3_G.describe())
df_2.to_csv("my_description.csv")

df = pd.read_csv('my_description.csv')



color = {
    'background': '#111111',
    'text': '#7FDBFF'

}
app = dash.Dash(__name__, prevent_initial_callbacks=True)

app.config['suppress_callback_exceptions'] = True

colors=["k","b","c","r","g","m","yelow","pink","violet","brown","beige","gold","darkgreen","skyblue","darkblue","dimgray","lightgray","yelowgreen","navy","coral","dodgerblue","salmon","y","plum","purple"]

server = app.server

app.layout = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])


if __name__ == '__main__':
    app.run_server(debug=True)