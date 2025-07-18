import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tokenizer import tokenize
from sklearn.metrics import RocCurveDisplay, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Load model

with open('name_type_pipeline.pkl', 'rb') as file:
        loaded_pipeline = pickle.load(file)

with open('target_pipeline.pkl', 'rb') as file:
        loaded_target_pipeline = pickle.load(file)

test_y_name_type = pd.read_csv('test_y_name_type.csv', usecols=['cat'])
train_y_name_type = pd.read_csv('train_y_name_type.csv', usecols=['cat'])
dt = np.dtype([('val', np.float64, 28)])
name_type_test_pred = np.fromfile('name_type_test_pred.dat', dtype=dt)['val']
name_type_train_pred = np.fromfile('name_type_train_pred.dat', dtype=dt)['val']

types = pd.read_csv('types.csv', usecols=['type'])

mapping = pickle.load(open('label_mapping.pickle','rb'))

# Scores

st.write('f1_score train: ', f1_score(train_y_name_type, name_type_train_pred.argmax(-1), average='micro'))
st.write('f1_score test: ', f1_score(test_y_name_type, name_type_test_pred.argmax(-1), average='micro'))

st.write('roc_auc_score train: ', roc_auc_score(train_y_name_type, name_type_train_pred, multi_class='ovr'))
st.write('roc_auc_score test: ', roc_auc_score(test_y_name_type, name_type_test_pred, multi_class='ovr'))


# Plot!

label_binarizer_type = LabelBinarizer().fit(train_y_name_type)
y_onehot_test_type = label_binarizer_type.transform(test_y_name_type)

display = RocCurveDisplay.from_predictions(
    y_onehot_test_type.ravel(),
    name_type_test_pred.ravel(),
    name="micro-average OvR",
    plot_chance_level=True,
    despine=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Micro-averaged One-vs-Rest\nReceiver Operating Characteristic",
)

st.plotly_chart(display.figure_)


### Test models

text = st.text_input('Enter text')

if text:
    pred = pd.Series([text], name='data')

    proba = loaded_pipeline.predict_proba(pred)

    id = proba[0].argmax()

    answer = mapping.get(id, 'None')

    st.write('Category is ', answer)

### target model

st.text("""Example: 
        Интерьерная картина Кирпичи в руинах древней цивилизации на холсте 240х90 см. с подвесами Дом и сад Картина
        Ножницы Волна , 9", 23 см, шаг - 18 мм, цвет чёрный Хобби и творчество Набор для шитья одежды Пирог""")

name = st.text_input('Enter name')

type = st.selectbox(
    'Choose type',
    types
)

if name and type:
       pred = pd.Series(name + type, name='data')

       proba = loaded_target_pipeline.predict_proba(pred)

       target = proba[0].argmax()

       st.write('Target is ', target)
