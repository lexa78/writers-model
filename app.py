

import streamlit as st
from prepare import process
import numpy as np
import pickle

CLASS_LIST_FILE_NAME = 'class_list.pickle'
st.title('AI recognition writers demo')

txt_file = st.file_uploader('Load an text', type=['txt'])  

if not txt_file is None:                                          
    st.markdown('<h2>Source text</h2>', unsafe_allow_html=True)

    text_val=[]
    with open(txt_file, 'r', encoding="windows-1251") as f:
      text = f.read()
      text = text.replace('
', ' ')
      text_val.append(text)
      st.text(text_val)

    pred = process(text_val)
    st.markdown('<h2>Результат распознавания:</h2>', unsafe_allow_html=True)

    r = np.argmax(pred, axis=1)
    unique, counts = np.unique(r, return_counts=True)
    counts = counts/pred.shape[0]*100

    st.markdown('<ul>', unsafe_allow_html=True)
    with open(CLASS_LIST_FILE_NAME, 'rb') as handle:
        CLASS_LIST = pickle.load(handle)
    for i in range(len(unique)):
      st.markdown('<b>{:10s}</b> - <span style="color:blue;">{:<.2f}%</span>'.format(CLASS_LIST[unique[i]], counts[i]), unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)
