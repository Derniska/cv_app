#!/usr/bin/env python
# coding: utf-8

import os
import io
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from tqdm.auto import tqdm
import datetime
import transformers
from transformers import pipeline
import torch
####################################
########## YOUR CODE HERE ##########
####################################
# You will need to download a model
# to implement summarization from 
# HugginFace Hub.
#
# You may want to use following models:
# https://huggingface.co/Falconsai/text_summarization
# https://huggingface.co/knkarthick/MEETING_SUMMARY
# ...or any other you like, but think of 
# the size of the model (<1GB recommended)
#
# Your code may look like this:
#from transformers import pipeline
#with st.spinner('Please wait, application is initializing...'):
#    MODEL_SUM_NAME = '<YOUR_MODEL>'
#    SUMMARIZATOR = pipeline("summarization", model=MODEL_SUM_NAME)
####################################

# page headers and info text
st.set_page_config(
    page_title='Text extraction', 
    page_icon=':microscope:'
)
st.sidebar.header('Extract and summarize text from an Image or a PDF file')
st.header('Text extraction and summarization', divider='rainbow')

st.markdown(
    f"""
    Extract text from an image or a PDF file and summarize it, if you want to
    """
)
st.divider()

####################################
########## YOUR CODE HERE ##########
####################################
def ocr_pdf_file(file, lang = 'eng'):
    if file.type == 'application/pdf':
        bytes_file = file.read()
        images = convert_from_bytes(bytes_file,
                                   fmt = "JPEG",
                                   dpi = 200)
        text = ''
        for image in tqdm(images):
            text_tmp = str(
                pytesseract.image_to_string(
                    image,
                    lang = lang
                )
            )
            text = ''.join([text, text_tmp])
        return text
    elif file.type.startswith('image/'):
        image = Image.open(file)
        text_image = pytesseract.image_to_string(
            image,
            lang = lang
        )
        if text_image == "":
            return 'No text found on image'
        else:
            return text_image
    else:
        return f'Unsupported file type: {file.type}'
        
def summarize_text(text , max_tokens, min_tokens):
    pipe = pipeline(
                "text2text-generation",
                model="google/t5gemma-b-b-prefixlm",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            
    prompt = f""""Summarize text in key points {text}"""
    outputs = pipe(
        prompt,
        max_new_tokens= max_tokens,           
        min_new_tokens=min_tokens,            
        repetition_penalty=2.5,       
        length_penalty= 1.9,           
        num_beams=4,                  
        early_stopping=True,          
        temperature=0.7,              
        do_sample=True,                
    )
    summary = outputs[0]["generated_text"]
    return summary


st.write('Upload your image or PDF file')
uploaded_file = st.file_uploader('Select PDF file or image (JPG format)')

if uploaded_file is not None:
    st.write(f"File type: {uploaded_file.type}")
else:
    st.write("No file uploaded yet")
lang = st.selectbox(
    "Choose the language of the file",
    ('eng', "rus", "eng+rus")
)
if uploaded_file is not None:
    with st.spinner('Please wait...'):
        ocr_text = ocr_pdf_file(uploaded_file, lang=lang)
        st.write(ocr_text)
        
        if "No text" not in ocr_text and "Unsupported file type" not in ocr_text:
            msg = '{} - the content of the file "{}" was successfully recognized\n'.format(
                datetime.datetime.now(),
                uploaded_file.name,
            )
        else:
            msg = '{} - the content of the file "{}" was not recognized\n'.format(
                datetime.datetime.now(),
                uploaded_file.name,
            )
        with open('history.log', 'a') as file:
            file.write(msg)
            
    st.markdown("--" *20)
    if st.button('Summarize text'):
        with st.spinner("Summarizing your text. It may take up to 30 seconds ..."):
            try:
                max_tokens = int(len(ocr_text.split(' ')) * 0.25)
                min_tokens = int(len(ocr_text.split(' ')) * 0.1)
                summary = summarize_text(ocr_text, max_tokens, min_tokens)
                st.subheader("Summarization result:")
                st.success(summary)
                summary_msg = f'{datetime.datetime.now()} - File "{uploaded_file.name}" was summarized\n'
                
                with open('history.log', 'a') as file:
                    file.write(summary_msg)
            except Exception as e:
                            st.error(f"Summarization error: {e}")

# Use example from the class with
# OCR model for text extracting from 
# the image or PDF file.
#
# Do not forget to add text summarization 
# model and display the output to the OCR 
# application's page  
####################################