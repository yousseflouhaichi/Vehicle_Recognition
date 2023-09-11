import streamlit as st
from predict import *
import pandas as pd
import torch
import tensorflow as tf
from PIL import Image



st.set_page_config(page_title="Car Recognition", page_icon=":car:")
st.markdown("<h1 style='text-align: center; color: #31333F; font-size: 60px;'>Vehicle Recognition</h1>", unsafe_allow_html=True)

@st.cache_resource(ttl=3600)
def read_list_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return [item.strip() for item in content]

MakeModelClasses = read_list_from_file('assets/classes/MakeModelClasses.txt')
TypeClasses = read_list_from_file('assets/classes/TypeClasses.txt')
ColorClasses = read_list_from_file('assets/classes/ColorClasses.txt')




# Load the saved models
@st.cache_resource(show_spinner='Loading models, please wait...', ttl=3600)
def load_model(PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(PATH, map_location=device)
    model.eval()
    model.to(device)
    return model

model = load_model("./models/inception_v3_model.pt")
color_model = load_model("./models/car_color_model.pt")
type_model = load_model("./models/car_type_model.pt")


file = st.file_uploader("Upload Your Image")

if file:
    image = Image.open(file)
    st.image(file, use_column_width=True)
    predicted_class = MakeModelPredict(model, file, MakeModelClasses)[0]
    predicted_color = ColorPredict(color_model, file, ColorClasses)[0]
    predicted_type = TypePredict(type_model, file, TypeClasses)[0]

    st.markdown("</br>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #31333F; font-size: 20px;'>Predicted features</h5>", unsafe_allow_html=True)

    result_dict = {
        'Make & Model': [predicted_class],
        'Type': [predicted_type],
        'Color': [predicted_color],
    }
    result_df = pd.DataFrame(result_dict)

    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    hide_table_col_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    # Display a static table
    st.table(result_df)


#Hide menu and footer
st.markdown("""
<style>
   #MainMenu, footer {visibility: hidden;}
</style>
""",unsafe_allow_html=True)
