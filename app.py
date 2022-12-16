from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb



def main() :
    
    st.title('K-means 클러스터링')
    
    file = st.file_uploader('CSV파일 업로드', type=['csv'])
    
    if file is not None :
        df = pd.read_csv(file)
        st.dataframe(df)
    
    
    column_list = df.columns
    
    selected_columns = st.multiselect('X로 사용할 컬럼을 선택하세요', column_list)
    
    X = df[selected_columns]
    
    st.dataframe(X)
    
    st.subheader('WCSS를 위한 클러스터링 갯수를 선택')
    max_number = st.slider('최대 그룹 선택', 2, 20, value=10)
    
    wcss = []
    for k in np.arange( 1, max_number+1 ) :
        kmeans = KMeans(n_clusters= k , random_state= 5 )
        kmeans.fit( X )
        wcss.append( kmeans.inertia_ )
    
    # st.write(wcss)
    
    fig1 = plt.figure()
    x = np.arange(1, max_number+1)
    plt.plot(x, wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    st.pyplot(fig1)
    
    
    # 실제로 그룹핑할 갯수 선택!
    # k = st.slider('그룹 갯수 결정', 1, max_number)
    
    k = st.number_input('그룹 갯수 결정', 1, max_number)
    
    kmeans = KMeans(n_clusters= k, random_state=5)
    
    y_pred = kmeans.fit_predict(X)
    
    df['Group'] = y_pred
    
    st.dataframe(df.sort_values('Group'))
    
    df.to_csv('result.csv')
    

if __name__ == '__main__' :
    main()