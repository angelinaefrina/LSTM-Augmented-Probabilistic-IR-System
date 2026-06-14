import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dot

def build_lstm_proximity_model(vocab_size, embedding_dim=100, max_len=200):
    # Input Layers for Query and Document
    query_input = Input(shape=(max_len,), name="query_sequence")
    doc_input = Input(shape=(max_len,), name="doc_sequence")
    
    # Shared Embedding Layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
    
    query_emb = embedding_layer(query_input)
    doc_emb = embedding_layer(doc_input)
    
    # Shared LSTM Layer
    lstm_layer = LSTM(64, return_sequences=False, name="dynamic_dependency_lstm")
    
    query_context = lstm_layer(query_emb)
    doc_context = lstm_layer(doc_emb)
    
    # Proximity Scoring
    # Calculate the semantic proximity between the query and document context states
    proximity_score = Dot(axes=1, normalize=True, name="cosine_proximity")([query_context, doc_context])
    
    # Dense layer to map the score between 0 and 1
    output = Dense(1, activation='sigmoid', name="final_score")(proximity_score)
    
    model = Model(inputs=[query_input, doc_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model