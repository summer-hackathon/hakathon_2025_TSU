import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tokenizer import tokenize

# Load models

uri_classification = 'classification/'

with open(f'{uri_classification}name_type_pipeline.pkl', 'rb') as file:
        loaded_cat_pipeline = pickle.load(file)

with open(f'{uri_classification}target_pipeline.pkl', 'rb') as file:
        loaded_target_pipeline = pickle.load(file)

types = pd.read_csv(f'{uri_classification}types.csv', usecols=['type'])

cat_mapping = pickle.load(open(f'{uri_classification}label_mapping.pickle','rb'))
cat_dict = pickle.load(open(f'{uri_classification}cat_dict.pickle','rb'))


def load_available_types(data, types=types['type'], n_count=4):
    ret = []
    for tt in types:
        pred = pd.Series(' '.join([data, tt]), name='data')

        proba = loaded_target_pipeline.predict_proba(pred)

        target = proba[0].argmax()

        if target == 1:
             ret.append(tt)
        
        if len(ret) == n_count:
             break
    return ret
             


### Test models

st.text("""Example: 
        Интерьерная картина на холсте 240х90 см. с подвесами
        Ножницы Волна , 9", 23 см, шаг - 18 мм, цвет чёрный Набор для шитья одежды """)

name = st.text_input('Enter text')

if name:
    pred = pd.Series([name], name='data')

    proba = loaded_cat_pipeline.predict_proba(pred)

    id = proba[0].argmax()

    category = cat_mapping.get(id, 'None')

    st.write(f'Category is {category}')

    available_subcats = cat_dict[category]

    subcat = st.selectbox(
        'Choose sub category',
        available_subcats,
        index=None,
    )

### target model

type = st.selectbox(
    'Choose type',
    types,
    index=None,
)

st.write(f'Type is {type}')

if name and type and category:
    pred = pd.Series(' '.join([name, category, (subcat if subcat else ''), type]), name='data')

    proba = loaded_target_pipeline.predict_proba(pred)

    target = proba[0].argmax()

    st.write(f'Target is {target}')

    if target == 0:
        available_types = load_available_types(' '.join([name, category, (subcat if subcat else ''), subcat]))

        st.text(f"""Available types: {available_types} """)

        new_type = st.selectbox(
            'Available types',
            available_types,
            index=None
        )

        if name and new_type:
            new_pred = pd.Series(' '.join([name, category, (subcat if subcat else ''), new_type]), name='data')

            new_proba = loaded_target_pipeline.predict_proba(new_pred)

            new_target = new_proba[0].argmax()

            st.write(f'Target is {new_target} now')

