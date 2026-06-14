import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

class IRPreprocessor:
    def __init__(self, max_seq_length=200, max_vocab_size=20000):
        self.max_seq_length = max_seq_length
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")

    def fit(self, all_texts):
        """Fits the tokenizer on the entire corpus (documents + queries)."""
        self.tokenizer.fit_on_texts(all_texts)

    def vectorize_and_pad(self, texts):
        """Converts text to integer sequences and pads them."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        # padding='post' ensures the LSTM processes the actual words first before hitting zeroes
        padded = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post', truncating='post')
        return padded

    def tokenize_for_bm25(self, text):
        """Standard word tokenization for the base BM25 model."""
        return word_tokenize(text.lower())