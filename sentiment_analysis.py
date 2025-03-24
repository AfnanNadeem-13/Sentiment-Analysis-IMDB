import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the dataset
file_path = "train.csv"  # Adjust if needed
df = pd.read_csv(file_path)

# Convert sentiment labels to numerical values
df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})

# Load stopwords once (optimization)
stop_words = set(stopwords.words('english'))

# Function to clean text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize text
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Apply preprocessing
df['text'] = df['text'].apply(preprocess_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Convert text into numerical representation using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Predict sentiment on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display sample predictions
sample_texts = X_test[:5].tolist()  # Convert to list for iteration
sample_predictions = model.predict(vectorizer.transform(sample_texts))

for text, prediction in zip(sample_texts, sample_predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Review: {text}\nPredicted Sentiment: {sentiment}\n")
