import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def run_eda_app() :
    
    file = st.file_uploader('CSV파일 업로드', type=['csv'])