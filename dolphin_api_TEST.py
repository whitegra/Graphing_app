# -*- coding: utf-8 -*-
import numpy as np
import dolphin
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown

# -----------------------------
#  1. Load & Tokenize a Smaller Dataset
# -----------------------------
nltk.download("brown")
nltk.download("punkt")

sentences = brown.sents()[:500]  # Use only the first 500 sentences for faster training
sentences = [" ".join(sent) for sent in sentences]  # Convert word lists into sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]  # Tokenize

# -----------------------------
# 2. Build a Smaller Vocabulary (Top 5000 Words)
# -----------------------------
all_words = [word for sent in tokenized_sentences for word in sent]
word_freq = {word: all_words.count(word) for word in set(all_words)}
sorted_vocab = sorted(word_freq, key=word_freq.get, reverse=True)[:1000]  #  Keep only top 5000 words

word_to_index = {word: i for i, word in enumerate(sorted_vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(word_to_index)

print(f"vocabulary size: {vocab_size}")

# -----------------------------
# 3. Prepare Training Data (Next-Word Prediction)
# -----------------------------
seq_length = 4  # Reduce sequence length for faster training
x_train, y_train = [], []

for sent in tokenized_sentences:
    indices = [word_to_index[word] for word in sent if word in word_to_index]
    for i in range(len(indices) - seq_length):
        x_train.append(indices[i:i + seq_length])  # Input sequence
        y_train.append(indices[i + seq_length])  # Target word

# Convert to `dolphin.Tensor`
x_train = dolphin.Tensor(np.array(x_train))
y_train = dolphin.Tensor(np.array(y_train))

batch_size = 8  # Batch Size
embed_dim = 32  # embedding size
num_heads = 2   # attention heads
hidden_dim = 64  # hidden layer size
num_layers = 2  # 2 Transformer layers
epochs = 20  # epochs
lr = 0.001

# -----------------------------
#  4. Convert Tokens to Embeddings
# -----------------------------
embedding_weights = dolphin.Tensor(np.random.randn(vocab_size, embed_dim) * 0.01, requires_grad=True)
embedding_weights.grad = np.zeros_like(embedding_weights.data)  #  Ensure gradients start at zero

x_embed = embedding_weights.data[x_train.data.astype(int)]
x_embed = dolphin.Tensor(x_embed, requires_grad=True)

print("Fixed x_embed shape:", x_embed.data.shape)  # Should be (num_samples, seq_length, embed_dim)

# -----------------------------
# 5. Build Transformer Encoder Model
# -----------------------------
encoder_blocks = [dolphin.TransformerEncoderBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]

def transformer_forward(x):
    """ Forward pass through multiple encoder blocks. """
    for block in encoder_blocks:
        x = block(x)
    return x.matmul(embedding_weights.transpose((1, 0)))  # Correct projection(check)

# -----------------------------
#  6. Initialize Optimizer
# -----------------------------
optimizer = dolphin.Adam(
    parameters=[embedding_weights] +  
    [p for block in encoder_blocks for p in vars(block).values() if isinstance(p, dolphin.Tensor)],  # Only Tensors
    lr=lr
)

# -----------------------------
# 7. Training Loop (Optimized for Speed)
# -----------------------------
print("\n🚀 Training Transformer Model...\n")
for epoch in range(epochs):
    logits = transformer_forward(x_embed)
    loss = dolphin.cross_entropy_loss(logits, y_train)

    loss.backward()

    optimizer.step()  # Reset gradients

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data:.4f}")

print("\nTraining complete!\n")

# -----------------------------
# 8. Generate Text!
# -----------------------------
def generate_text(start_word, length=5):
    """ Generate text using the trained Transformer model. """
    words = [start_word]
    
    for _ in range(length):
        token = dolphin.Tensor([[word_to_index.get(words[-1], 0)]])  # Convert last word to tensor
        
        embed = embedding_weights.data[token.data.astype(int)].reshape(1, -1)  # Lookup word embedding
        embed = dolphin.Tensor(embed, requires_grad=False)  # Convert back to Tensor
        
        logits = transformer_forward(embed)  # Pass through Transformer
        probs = dolphin.softmax(logits)  # Convert logits to probabilities
        next_word_index = np.random.choice(len(sorted_vocab), p=probs.data.flatten())  # Sample randomly

        words.append(index_to_word[next_word_index])  # Convert index to word

    return " ".join(words)

# -----------------------------
#  9. Show Results!
# -----------------------------
print("\n AI-Generated Sentences:")
print(f"{generate_text('The government', length=6)}")
print(f"{generate_text('Artificial Intelligence is', length=6)}")
print(f" {generate_text('Science proves that', length=6)}")
