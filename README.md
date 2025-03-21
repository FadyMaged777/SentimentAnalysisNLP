# Sentiment Analysis with LSTM on IMDB Reviews

This project implements a Long Short-Term Memory (LSTM) neural network for binary sentiment classification using TensorFlow/Keras.

## Dataset
- **IMDB Movie Review Dataset** (50,000 reviews)
  - 25,000 training samples
  - 25,000 testing samples
  - Binary labels: Positive (1) / Negative (0)
- Vocabulary limited to top 10,000 words
- Sequences padded/truncated to 500 tokens

## Model Architecture
```python
```
Sequential(
    Embedding(input_dim=10000, output_dim=32, input_length=500),
    LSTM(32),
    Dense(1, activation='sigmoid')
)

Optimizer: Adam

Loss Function: Binary Cross-Entropy

Training: 10 epochs with 128 batch size

Results
Test Accuracy: ~85%

Training Visualization:

Accuracy/Loss curves for training and validation sets

Output:

85%
