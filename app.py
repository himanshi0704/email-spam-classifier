import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Stemmer
ps = PorterStemmer()

# Function for text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Stemming words

    return " ".join(y)

# Load saved models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI Improvements
st.set_page_config(page_title="Email Spam Classifier", page_icon="📩", layout="centered")

# Custom Styling for Better Visibility
st.markdown(
    """
    <style>
        /* Background Styling */
        .main {
            background-color: #f8f9fa;
        }
        /* Title Styling */
        h1 {
            color: #003366 !important;
            text-align: center;
            font-weight: bold;
        }
        /* Subtitle & Footer Styling */
        .subtitle, .footer {
            color: #333333 !important;
            font-size: 14px;
            text-align: center;
        }
        /* Text Area Styling */
        .stTextArea label {
            color: #003366 !important;
            font-size: 16px;
            font-weight: bold;
        }
        /* Button Styling */
        .stButton button {
            background-color: #004080 !important;
            color: white !important;
            font-size: 16px !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        .stButton button:hover {
            background-color: #0066cc !important;
        }
        /* Spam & Not Spam Result Styling */
        .spam {
            background-color: #cc0000 !important;
            color: white !important;
            font-weight: bold !important;
            text-align: center;
            font-size: 24px !important;
            padding: 10px;
            border-radius: 8px;
        }
        .not-spam {
            background-color: #008000 !important;
            color: white !important;
            font-weight: bold !important;
            text-align: center;
            font-size: 24px !important;
            padding: 10px;
            border-radius: 8px;
        }
        .spam-message, .safe-message {
            font-size: 18px !important;
            font-weight: bold !important;
            text-align: center;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .spam-message {
            background-color: #ffcccc !important;
            color: #660000 !important;
        }
        .safe-message {
            background-color: #ccffcc !important;
            color: #006600 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("📩 Email Spam Classifier")

# Input Text Box
st.markdown("<p class='subtitle'>🔹 Enter the email message below to classify it as SPAM or NOT SPAM.</p>", unsafe_allow_html=True)
input_sms = st.text_area("✍️ Enter the message:", height=150)

# Predict Button with Styling
if st.button("🚀 Predict", help="Click to classify the message"):

    if input_sms.strip() == "":
        st.markdown(
            """
            <div style="
                background-color: #fff3cd; 
                color: #856404; 
                padding: 12px; 
                border: 2px solid #ffeeba; 
                border-radius: 8px;
                font-weight: bold; 
                font-size: 18px;
                text-align: center;">
                ⚠️ Please enter a message before predicting!
            </div>
            """, unsafe_allow_html=True
        )
    else:
        # Preprocess text
        transformed_sms = transform_text(input_sms)
        # Vectorize text
        vector_input = tfidf.transform([transformed_sms])
        # Predict spam or not
        result = model.predict(vector_input)[0]

        # Display Result with Better Visibility
        if result == 1:
            st.markdown("<h2 class='spam'>🚨 SPAM!</h2>", unsafe_allow_html=True)
            st.markdown("<p class='spam-message'>⚠️ This message is classified as <b>SPAM</b>. Be cautious!</p>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 class='not-spam'>🔰NOT SPAM</h2>", unsafe_allow_html=True)
            st.markdown("<p class='safe-message'>✅ This message is classified as <b>SAFE</b>.</p>", unsafe_allow_html=True)

