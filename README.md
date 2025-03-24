Sentiment Analysis on IMDB Reviews

📌 Project Overview

This project performs Sentiment Analysis on IMDB movie reviews using Natural Language Processing (NLP) techniques. The goal is to classify reviews as Positive or Negative based on their text content.

📂 Dataset

We used the IMDB 50K Movie Reviews Dataset from Kaggle:
🔗 Dataset Link

The dataset consists of:

train.csv: Training data (used to train the model)

test.csv: Testing data (used to evaluate the model)

🛠️ Installation

Before running the project, install the required dependencies:

pip install -r requirements.txt

Required Python libraries:

nltk

numpy

pandas

scikit-learn

matplotlib

seaborn

If you don’t have requirements.txt, install manually:

pip install nltk numpy pandas scikit-learn matplotlib seaborn

🚀 How to Run

Clone the repository:

git clone https://github.com/AfnanNadeem-13/Sentiment-Analysis-IMDB.git
cd Sentiment-Analysis-IMDB

Run the sentiment analysis script:

python sentiment_analysis.py

📊 Model Performance

The trained model achieved the following results:

Class

Precision

Recall

F1-score

Support

Negative

0.89

0.87

0.88

2506

Positive

0.87

0.89

0.88

2494

Overall Accuracy: 87.78%

🔍 Data Visualizations

Word Cloud of Positive Reviews



Word Cloud of Negative Reviews



Confusion Matrix



📜 File Structure

Sentiment-Analysis-IMDB/
│── sentiment_analysis.py    # Main script for training/testing the model
│── train.csv                # Training dataset
│── test.csv                 # Testing dataset
│── visuals/                 # Folder for storing visualizations
│── README.md                # Project documentation
│── requirements.txt         # Dependencies

🤝 Contributing

Feel free to contribute by improving the model, adding new features, or enhancing visualizations!

🏆 Acknowledgments

NLTK for text preprocessing

Scikit-Learn for machine learning models

Matplotlib & Seaborn for visualizations

Kaggle for providing the dataset
