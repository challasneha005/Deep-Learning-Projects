 # Fake News Classifier

## Overview
This project implements a Fake News Classifier using Natural Language Processing (NLP) and a Long Short-Term Memory (LSTM) neural network. The model is trained on a dataset of news headlines to classify them as real or fake.

## Dataset
Download the dataset [Fake News Dataset (Kaggle)](https://www.kaggle.com/code/ahmedtronic/fake-news-classification/input?select=train.csv)
The model uses a dataset stored in `train.csv` with the following structure:
- `title`: The headline of the news article.
- `label`: The classification label (1 for fake news, 0 for real news).

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install pandas numpy nltk tensorflow scikit-learn
```

## Data Preprocessing
- The dataset is loaded and missing values are removed.
- Text preprocessing includes:
  - Removing non-alphabetic characters
  - Converting text to lowercase
  - Removing stopwords
  - Applying stemming using PorterStemmer
- The cleaned text is converted into numerical representations using one-hot encoding and embedded with TensorFlow’s `Embedding` layer.

## Model Architecture
- **Embedding Layer**: Converts text input into dense vectors.
- **LSTM Layer**: Captures long-term dependencies in text sequences.
- **Dropout Layers**: Prevents overfitting.
- **Dense Output Layer**: Uses a sigmoid activation function for binary classification.

## Training and Evaluation
- The dataset is split into training and testing sets (67%-33%).
- The model is trained using binary cross-entropy loss and the Adam optimizer.
- The model is evaluated using:
  - Confusion Matrix
  - Accuracy Score

## Running the Model
1. Load the dataset and preprocess it.
2. Train the LSTM model using:
```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
```
3. Evaluate the model:
```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Confusion Matrix:", cm)
print("Accuracy:", acc)
```

## Future Improvements
- Experiment with different embedding techniques (e.g., Word2Vec, GloVe).
- Use bidirectional LSTM for improved performance.
- Fine-tune hyperparameters for better accuracy.

 🤝 Contributing

Feel free to open an issue or submit a pull request!
