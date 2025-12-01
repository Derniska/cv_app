#!/usr/bin/env python
# coding: utf-8

import os
import io
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

####################################
# ######### YOUR CODE HERE ##########
# ###################################
# 
# Wait until NLP part is finished.
#
# ###################################

# page headers and info text
st.set_page_config(
    page_title='<YOUR_TEXT_HERE>', 
    page_icon=':microscope:'
)
st.sidebar.header('<YOUR_HEADER_HERE>')
st.header('<YOUR_HEADER_HERE>', divider='rainbow')

st.markdown(
    f"""
    <YOUR_DESCRIPTION_HERE>
    """
)
st.divider()

####################################
# ######### YOUR CODE HERE ##########
# ###################################
# 
# Wait until NLP part is finished.
#
# ###################################
