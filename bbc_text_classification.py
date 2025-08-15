#!/usr/bin/env python3
# Import libraries
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import joblib
from collections import Counter
from textblob import Word
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, \
    recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Activation, Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Flatten, GRU, Conv1D, \
    MaxPooling1D, Bidirectional
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import urllib
import requests
import re
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    sns.set()

    # Download NLTK data
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('gutenberg')
    nltk.download('brown')
    nltk.download("reuters")
    nltk.download('words')

    # Load data - try different common filenames
    possible_files = [
        "bbc-text.csv",
        "bbc-text-large.csv",
        "BBC News Train.csv",
        "bbc_text.csv",
        "BBC_News_Train.csv",
        "bbc-fulltext-and-category.csv",
        "bbc_dataset.csv"
    ]

    df = None
    found_file = None

    for filename in possible_files:
        try:
            df = pd.read_csv(filename, engine='python', encoding='UTF-8')
            found_file = filename
            print(f"Found dataset: {filename}")
            break
        except FileNotFoundError:
            continue

    if df is None:
        print("Error: BBC dataset not found!")
        print("Please run: python download_large_datasets.py first")
        exit(1)

    print(f"Dataset shape: {df.shape}")
    print("Categories found:", df['category'].unique())
    print("Category counts:")
    print(df['category'].value_counts())

    # Preprocessing
    df['text'] = df['text'].fillna("")
    print(df.isna().sum())
    df['lower_case'] = df['text'].apply(lambda x: x.lower().strip().replace('\n', ' ').replace('\r', ' '))

    df['alphabatic'] = df['lower_case'].apply(lambda x: re.sub(r'[^a-zA-Z\']', ' ', x)).apply(
        lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
    df['without-link'] = df['alphabatic'].apply(lambda x: re.sub(r'http\S+', '', x))

    tokenizer = RegexpTokenizer(r'\w+')
    df['Special_word'] = df.apply(lambda row: tokenizer.tokenize(row['lower_case']), axis=1)

    stop = [word for word in stopwords.words('english') if
            word not in ["my", "haven't", "aren't", "can", "no", "why", "through", "herself", "she", "he", "himself",
                         "you", "you're", "myself", "not", "here", "some", "do", "does", "did", "will", "don't",
                         "doesn't", "didn't", "won't", "should", "should've", "couldn't", "mightn't", "mustn't",
                         "shouldn't", "hadn't", "wasn't", "wouldn't"]]

    df['stop_words'] = df['Special_word'].apply(lambda x: [item for item in x if item not in stop])
    df['stop_words'] = df['stop_words'].astype('str')

    df['short_word'] = df['stop_words'].str.findall(r'\w{2,}')
    df['string'] = df['short_word'].str.join(' ')

    df['Text'] = df['string'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # Visualizations
    fig = plt.figure(figsize=(14, 7))
    df['length'] = df.text.str.split().apply(len)
    ax1 = fig.add_subplot(122)
    sns.histplot(df['length'], ax=ax1, color='green')
    describe = df.length.describe().to_frame().round(2)

    ax2 = fig.add_subplot(121)
    ax2.axis('off')
    font_size = 14
    bbox = [0, 0, 1, 1]
    table = ax2.table(cellText=describe.values, rowLabels=describe.index, bbox=bbox, colLabels=describe.columns)
    table.set_fontsize(font_size)
    fig.suptitle('Distribution of text length for text.', fontsize=16)
    plt.show()

    sns.set_theme(style="whitegrid")
    sns.countplot(x=df["category"])
    plt.show()

    # Skip detailed visualizations that cause errors - go straight to ML
    print("Skipping detailed word visualizations to avoid DataFrame issues...")
    print("Proceeding to machine learning models...")

    # Machine Learning Models
    x_train, x_test, y_train, y_test = train_test_split(df["Text"], df["category"], test_size=0.25, random_state=42)
    count_vect = CountVectorizer(ngram_range=(1, 2))
    transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    x_train_counts = count_vect.fit_transform(x_train)
    x_train_tfidf = transformer.fit_transform(x_train_counts)

    x_test_counts = count_vect.transform(x_test)
    x_test_tfidf = transformer.transform(x_test_counts)

    print(x_train_tfidf.shape, x_test_tfidf.shape, y_train.shape, y_test.shape)

    joblib.dump(count_vect, 'count_vect.pkl')

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    lr.fit(x_train_tfidf, y_train)
    y_pred1 = lr.predict(x_test_tfidf)
    print("Accuracy: " + str(accuracy_score(y_test, y_pred1)))
    print(classification_report(y_test, y_pred1))

    scores = cross_val_score(lr, x_train_tfidf, y_train, cv=10)
    print("Cross-validated scores:", scores)
    joblib.dump(lr, 'Text_LR.pkl')

    # SVM
    print("Training SVM...")
    svc = LinearSVC()
    svc.fit(x_train_tfidf, y_train)
    y_pred2 = svc.predict(x_test_tfidf)
    print("Accuracy: " + str(accuracy_score(y_test, y_pred2)))
    print(classification_report(y_test, y_pred2))

    scores = cross_val_score(svc, x_train_tfidf, y_train, cv=10)
    print("Cross-validated scores:", scores)
    joblib.dump(svc, 'Text_SVM.pkl')

    # Naive Bayes
    print("Training Naive Bayes...")
    mnb = MultinomialNB()
    mnb.fit(x_train_tfidf, y_train)
    y_pred3 = mnb.predict(x_test_tfidf)
    print("Accuracy: " + str(accuracy_score(y_test, y_pred3)))
    print(classification_report(y_test, y_pred3))

    scores = cross_val_score(mnb, x_train_tfidf, y_train, cv=10)
    print("Cross-validated scores:", scores)

    # Random Forest
    print("Training Random Forest...")
    rfc = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
    rfc.fit(x_train_tfidf, y_train)
    y_pred4 = rfc.predict(x_test_tfidf)
    print("Accuracy: " + str(accuracy_score(y_test, y_pred4)))
    print(classification_report(y_test, y_pred4))

    scores = cross_val_score(rfc, x_train_tfidf, y_train, cv=10)
    print("Cross-validated scores:", scores)

    # Gradient Boosting
    print("Training Gradient Boosting...")
    gbc = GradientBoostingClassifier(n_estimators=100, max_features='auto', max_depth=4, random_state=1, verbose=1)
    gbc.fit(x_train_tfidf, y_train)
    y_pred5 = gbc.predict(x_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred5))
    print(classification_report(y_test, y_pred5))

    scores = cross_val_score(gbc, x_train_tfidf, y_train, cv=5)
    print("Cross-validated scores:", scores)

    # Ensemble Voting Classifier
    print("Training Ensemble...")
    mnb = MultinomialNB()
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=12, random_state=42)
    lr = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    svc = SVC(probability=True)
    ec = VotingClassifier(estimators=[('Multinominal NB', mnb), ('Random Forest', rfc), ('Logistic Regression', lr),
                                      ('Support Vector Machine', svc)], voting='soft', weights=[1, 2, 3, 4])
    ec.fit(x_train_tfidf, y_train)
    y_pred6 = ec.predict(x_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred6))
    print(classification_report(y_test, y_pred6))

    scores = cross_val_score(ec, x_train_tfidf, y_train, cv=10)
    print("Cross-validated scores:", scores)
    joblib.dump(ec, 'Text_Ensemble.pkl')

    # AdaBoost
    print("Training AdaBoost...")
    rfc = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
    abc = AdaBoostClassifier(base_estimator=rfc, learning_rate=0.2, n_estimators=100)
    abc.fit(x_train_tfidf, y_train)
    y_pred7 = abc.predict(x_test_tfidf)
    print("Accuracy: " + str(accuracy_score(y_test, y_pred7)))
    print(classification_report(y_test, y_pred7))

    scores = cross_val_score(abc, x_train_tfidf, y_train, cv=10)
    print("Cross-validated scores:", scores)

    # Model Comparison
    print("\nCreating model comparison...")
    try:
        comparison_data = {
            'Logistic Regression': [accuracy_score(y_test, y_pred1) * 100,
                                    f1_score(y_test, y_pred1, average='macro') * 100],
            'SVM': [accuracy_score(y_test, y_pred2) * 100, f1_score(y_test, y_pred2, average='macro') * 100],
            'Naive Bayes': [accuracy_score(y_test, y_pred3) * 100, f1_score(y_test, y_pred3, average='macro') * 100],
            'Random Forest': [accuracy_score(y_test, y_pred4) * 100, f1_score(y_test, y_pred4, average='macro') * 100],
            'Gradient Boosting': [accuracy_score(y_test, y_pred5) * 100,
                                  f1_score(y_test, y_pred5, average='macro') * 100],
            'Ensemble': [accuracy_score(y_test, y_pred6) * 100, f1_score(y_test, y_pred6, average='macro') * 100],
            'AdaBoost': [accuracy_score(y_test, y_pred7) * 100, f1_score(y_test, y_pred7, average='macro') * 100]
        }

        Comparison_unibi = pd.DataFrame(comparison_data, index=['Accuracy', 'F1_score'])
        print('Model Comparison:')
        print(Comparison_unibi)
    except Exception as e:
        print(f"Error in comparison: {e}")

    # Deep Learning Models
    print("\nStarting Deep Learning Models...")

    # Dynamic label encoding based on available categories
    unique_categories = sorted(df['category'].unique())
    num_classes = len(unique_categories)

    print(f"Encoding {num_classes} categories: {unique_categories}")

    # Create label mapping
    label_mapping = {category: idx for idx, category in enumerate(unique_categories)}
    print(f"Label mapping: {label_mapping}")

    # Apply label encoding
    df['LABEL'] = df['category'].map(label_mapping)

    vocabulary_size = 15000
    max_text_len = 768
    stemmer = SnowballStemmer('english')
    stop_words = [word for word in stopwords.words('english') if
                  word not in ["my", "haven't", "aren't", "can", "no", "why", "through", "herself", "she", "he",
                               "himself", "you", "you're", "myself", "not", "here", "some", "do", "does", "did", "will",
                               "don't", "doesn't", "didn't", "won't", "should", "should've", "couldn't", "mightn't",
                               "mustn't", "shouldn't", "hadn't", "wasn't", "wouldn't"]]


    def preprocess_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.lower().split()
        words = [stemmer.stem(word) for word in words if not word in stop_words]
        cleaned_text = ' '.join(words)
        return cleaned_text


    df['cleaned_text'] = df['text'].apply(preprocess_text)

    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(df['cleaned_text'].values)
    le = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {le}")
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'].values)
    X_DeepLearning = pad_sequences(sequences, maxlen=max_text_len)

    labels = to_categorical(df['LABEL'], num_classes=num_classes)
    XX_train, XX_test, y_train, y_test = train_test_split(X_DeepLearning, labels, test_size=0.25, random_state=42)
    print((XX_train.shape, y_train.shape, XX_test.shape, y_test.shape))

    # LSTM Model 1
    print("Training LSTM Model 1...")
    epochs = 25
    emb_dim = 256
    batch_size = 50
    model_lstm1 = Sequential()
    model_lstm1.add(Embedding(vocabulary_size, emb_dim, input_length=X_DeepLearning.shape[1]))
    model_lstm1.add(SpatialDropout1D(0.8))
    model_lstm1.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5)))
    model_lstm1.add(Dropout(0.5))
    model_lstm1.add(Flatten())
    model_lstm1.add(Dense(64, activation='relu'))
    model_lstm1.add(Dropout(0.5))
    model_lstm1.add(Dense(num_classes, activation='softmax'))
    model_lstm1.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

    checkpoint_callback = ModelCheckpoint(filepath="lastm-1-layer-best_model.h5", save_best_only=True,
                                          monitor="val_acc", mode="max", verbose=1)
    early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1,
                                            restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min",
                                           min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks = [checkpoint_callback, early_stopping_callback, reduce_lr_callback]

    history_lstm1 = model_lstm1.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size,
                                    validation_data=(XX_test, y_test), callbacks=callbacks)

    results_1 = model_lstm1.evaluate(XX_test, y_test, verbose=False)
    print(f'LSTM 1 Test results - Loss: {results_1[0]:.4f} - Accuracy: {100 * results_1[1]:.2f}%')

    # LSTM Model 2
    print("Training LSTM Model 2...")
    epochs = 20
    emb_dim = 120
    batch_size = 50
    model_lstm2 = Sequential()
    model_lstm2.add(Embedding(vocabulary_size, emb_dim, input_length=X_DeepLearning.shape[1]))
    model_lstm2.add(SpatialDropout1D(0.8))
    model_lstm2.add(Bidirectional(LSTM(200, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
    model_lstm2.add(Dropout(0.5))
    model_lstm2.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5)))
    model_lstm2.add(Dropout(0.5))
    model_lstm2.add(Flatten())
    model_lstm2.add(Dense(64, activation='relu'))
    model_lstm2.add(Dropout(0.5))
    model_lstm2.add(Dense(num_classes, activation='softmax'))
    model_lstm2.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

    callbacks2 = [
        ModelCheckpoint(filepath="lastm-2-layer-best_model.h5", save_best_only=True, monitor="val_acc", mode="max",
                        verbose=1),
        EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001,
                          cooldown=0, min_lr=0)
    ]

    history_lstm2 = model_lstm2.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size,
                                    validation_data=(XX_test, y_test), callbacks=callbacks2)

    results_2 = model_lstm2.evaluate(XX_test, y_test, verbose=False)
    print(f'LSTM 2 Test results - Loss: {results_2[0]:.4f} - Accuracy: {100 * results_2[1]:.2f}%')

    # GRU Model
    print("Training GRU Model...")
    epochs = 20
    emb_dim = 256
    batch_size = 50
    model_gru = Sequential()
    model_gru.add(Embedding(vocabulary_size, emb_dim, input_length=X_DeepLearning.shape[1]))
    model_gru.add(SpatialDropout1D(0.8))
    model_gru.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.2))
    model_gru.add(Dropout(0.5))
    model_gru.add(Dense(256, activation='relu'))
    model_gru.add(Dropout(0.5))
    model_gru.add(Dense(num_classes, activation='softmax'))
    model_gru.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

    callbacks3 = [
        ModelCheckpoint(filepath="gru-best_model.h5", save_best_only=True, monitor="val_acc", mode="max", verbose=1),
        EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001,
                          cooldown=0, min_lr=0)
    ]

    history_gru = model_gru.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(XX_test, y_test), callbacks=callbacks3)
    results_3 = model_gru.evaluate(XX_test, y_test, verbose=False)
    print(f'GRU Test results - Loss: {results_3[0]:.4f} - Accuracy: {100 * results_3[1]:.2f}%')

    # CNN + LSTM Model
    print("Training CNN+LSTM Model...")
    epochs = 20
    emb_dim = 256
    batch_size = 50
    model_cl = Sequential()
    model_cl.add(Embedding(vocabulary_size, emb_dim, input_length=X_DeepLearning.shape[1]))
    model_cl.add(SpatialDropout1D(0.8))
    model_cl.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
    model_cl.add(MaxPooling1D(pool_size=2))
    model_cl.add(Conv1D(filters=32, kernel_size=6, activation='relu'))
    model_cl.add(MaxPooling1D(pool_size=2))
    model_cl.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
    model_cl.add(Dropout(0.5))
    model_cl.add(Bidirectional(LSTM(400, dropout=0.5, recurrent_dropout=0.5)))
    model_cl.add(Dropout(0.5))
    model_cl.add(Flatten())
    model_cl.add(Dense(64, activation='relu'))
    model_cl.add(Dropout(0.5))
    model_cl.add(Dense(num_classes, activation='softmax'))
    model_cl.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    callbacks4 = [
        ModelCheckpoint(filepath="cnn+lastm-best_model.h5", save_best_only=True, monitor="val_acc", mode="max",
                        verbose=1),
        EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001,
                          cooldown=0, min_lr=0)
    ]

    history_cl = model_cl.fit(XX_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                              callbacks=callbacks4)

    results_4 = model_cl.evaluate(XX_test, y_test, verbose=False)
    print(f'CNN+LSTM Test results - Loss: {results_4[0]:.4f} - Accuracy: {100 * results_4[1]:.2f}%')

    print("\n" + "=" * 50)
    print("All models completed successfully!")
    print("Classical ML models and Deep Learning models trained!")
    print("=" * 50)