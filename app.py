# Import necessary libraries
import streamlit as st
import numpy as np 
import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set the page configuration
st.set_page_config(
    page_title="Bangla Aggressive Text Detection App",  # Title of the app displayed in the browser tab
    page_icon=":shield:",  # Path to a favicon or emoji to be displayed in the browser tab
    initial_sidebar_state="auto"  # Initial state of the sidebar ("auto", "expanded", or "collapsed")
)


# Load custom CSS styling
with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
     
# Function to load the pre-trained model
@st.cache_resource(experimental_allow_widgets=True)
def get_model():
    num_classes = 4  # Number of classes in the dataset
    model_name = "csebuetnlp/banglabert"  # Pre-trained model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained("Soyeda10/AggressiveBanglaBERT", num_labels=num_classes)
    return tokenizer, model

# Load the tokenizer and model
tokenizer, model = get_model()

# Dictionary to map predicted labels to their corresponding categories
dictionary_labels = {0: 'Religious Aggression', 1: 'Political Aggression', 2: 'Verbal Aggression', 3: 'Gendered Aggression'}

# Add a header to the Streamlit app
st.header("AggressiveTracker")

# Text area for user input
user_input = st.text_area("Enter your text here", height=300)

# Button for submitting the input
submit_button = st.button("Submit")

# Perform prediction when user input is provided and the submit button is clicked
if user_input and submit_button:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=250, return_tensors='pt')
    output = model(**test_sample)
    y_predicted = np.argmax(output.logits.detach().numpy(), axis=1)
    st.write("Prediction: ", dictionary_labels[y_predicted[0]])
