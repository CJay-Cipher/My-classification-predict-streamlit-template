"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
from nltk.stem import WordNetLemmatizer

# Vectorizer
news_vectorizer = open("resources/pipeline.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# DATA CLEANING
def text_cleaning(text):
    # Convert tweets to lowercase
    text = text.lower()

    # Tokenizing the tweets
    text = text.split(" ")
    
    # Initialize a lemmatizer object
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(i, pos='v') for i in text])
    
    return text

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		# Load your .pkl files into a dictionary
		models = {
			"Logistic regression": "lr.pkl",
			"Multinomial naive bayes": "mnb.pkl",
			"Linear SVC": "lr_svc.pkl"
		}
		# Create a selectbox to let the user choose the model
		model_selection = st.selectbox("Select Model", list(models.keys()))

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([text_cleaning(tweet_text)]).toarray()
		
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(f"resources/{models[model_selection]}"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

	