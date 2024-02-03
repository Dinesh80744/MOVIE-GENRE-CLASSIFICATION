# MOVIE-GENRE-CLASSIFICATION
This code is a Jupyter Notebook (`.ipynb` file) that contains a machine learning pipeline for predicting movie genres based on their descriptions. Let me break down what each section of the code does:

1. **Imports**: The code starts by importing necessary libraries such as pandas for data manipulation, numpy for numerical operations, seaborn and matplotlib for data visualization, nltk for natural language processing tasks, and various modules from scikit-learn for machine learning tasks.

2. **Data Loading**: The training and test data are loaded from files named "train_data.txt" and "test_data.txt" respectively, using pandas `read_csv` function.

3. **Data Exploration**: Some basic exploratory data analysis is performed on the training data, including checking the first few rows of the dataset, visualizing the distribution of movie genres, checking data summary statistics, and handling missing values.

4. **Data Cleaning**: The text data in the "description" column is cleaned using various preprocessing steps such as converting to lowercase, removing URLs, punctuation, stopwords, and other non-alphabetic characters. NLTK is used for tokenization and stemming.

5. **Feature Engineering**: TF-IDF vectorization is applied to convert the cleaned text data into numerical features.

6. **Model Training**: Several machine learning models including Logistic Regression, Naive Bayes, Decision Tree, and Support Vector Machine (SVM) are trained on the TF-IDF transformed training data.

7. **Model Evaluation**: The trained models are evaluated using classification metrics such as accuracy, precision, recall, and F1-score on a validation set. The accuracy of each model is printed out.

8. **Kaggle Submission**: Finally, there's an attempt to output the kernel results to a Kaggle kernel using the `!kaggle kernels output` command. This command is used to submit the results of the code execution to Kaggle.

Overall, this code performs a text classification task where movie genres are predicted based on the descriptions of the movies using various machine learning algorithms.
