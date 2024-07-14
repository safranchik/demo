import warnings

from dash.api import get_configs
from dash.utils import select_runs
from dash.figures import parameter_importance_figure, parallel_coordinates_figure, tsne_figure
import streamlit as st


warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

configs = get_configs(path='.')

col1, col2 = st.columns(2)

with st.sidebar:
    st.title("Cubyc Dash")
    st.write("A dashboard for hyperparameter optimization using Cubyc.")
    repo_path = st.text_input("Repository URL")
    token = st.text_input("Access Token")



with col1.container(border=False):
    runs, hyperparameters, metric = select_runs(path=repo_path)

with col2.container(border=False):
    parameter_importance_figure(runs, hyperparameters, metric)

    parallel_coordinates_figure(runs, hyperparameters, metric)

    # tsne_figure(runs, hyperparameters, metric)
