# install streamlit: pip install streamlit
# run: streamlit run app.py

import streamlit as st
import pickle
import time

# load the model
model = pickle.load(open('sentimental_analysis.pkl', 'rb'))

st.title('Twitter Sentiment Analysis')

tweet = st.text_input('Enter your tweet')

submit = st.button('Predict')

if submit:
    start = time.time()
    prediction = model.predict([tweet])
    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
    
    print(prediction[0])
    st.write(prediction[0])
    if __name__ == "__main__":
        st.write("Streamlit app is running. No need for app.run() in Streamlit.")
