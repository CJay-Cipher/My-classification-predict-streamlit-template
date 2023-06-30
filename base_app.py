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

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer


# DATA CLEANING & FEATURE ENGINEERING
def data_cleaning(data):
    copy_data = data.copy()

    # Convert tweets to lowercase
    copy_data['message'] = copy_data['message'].apply(lambda x: x.lower())

    # Tokenizing the tweets
    copy_data['message'] = copy_data['message'].apply(lambda x:[word for word in x.split(" ")])
    
    # Initialize a lemmatizer object
    lemmatizer = WordNetLemmatizer()
    copy_data['message'] = copy_data['message'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(i, pos='v') for i in x]))
    
    return copy_data


# FEATURE ENGINEERING
def feature_engineering():
    """ Setting the parameters for the Vectorizer """
    vectorizer = CountVectorizer(
        analyzer = 'word', 
        tokenizer = None, 
        preprocessor = None, 
        stop_words = None, 
        max_features = 180000,
        min_df = 1,
        ngram_range = (1, 2)
    )
    return vectorizer

# Vectorizer
vectorizer = feature_engineering()

# Load your raw data
raw = pd.read_csv("resources/train.csv")

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
		tweet_text = list(st.text_area("Enter Text","Type Here"))

		model_options = ["Logistic Regression", "Linear SVC", "Multinomial Naive Bayes"]
		model_selection = st.selectbox("Select Model", model_options)
		if model_selection == "Logistic Regression":
			selected_model = "lr"
		elif model_selection == "Linear SVC":
			selected_model = "lr_svc"
		else:
			selected_model = "mnb"

		if st.button("Classify"):
			# Transforming user input with vectorizer
			tweet_df = pd.DataFrame({"message": tweet_text})
			tweet_df = data_cleaning(tweet_df)
			vect_text = vectorizer.transform(tweet_df["message"])
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(f"pickle_files/{selected_model}.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
