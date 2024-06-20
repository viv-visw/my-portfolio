# Getting Started


## Overview

This page provides a step-by-step walkthrough of how you can use the Transformer model to predict the number of positive and negative movie reviews in a dataset of movie reviews.

The dataset you'll use for this task is the [IMDB Movie Review Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).


## Procedure


### Step 1: Download the dataset

Open a command-line Terminal window and enter the following code:

```bash
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```
  
This creates a directory named **aclImdb**, with the following structure:

```
aclImdb/
...train/
......pos/
......neg/
...test/
......pos/
......neg/
```

Here, the `train/pos/` directory contains a set of 12,500 text files, each of which contains the text body of a positive-sentiment movie review to be used as training data. The negative-sentiment reviews live in the `train/neg/` directories. In total, there are 25,000 text files for training and another 25,000 for testing.

Since this is a classification task, delete the `train/unsup` subdirectory:

```bash
!rm -r aclImdb/train/unsup
```


### Step 2: Prepare the dataset

Prepare a validation set by setting apart 20% of the training text files in a new directory, `aclImdb/val`:

```python
import os, pathlib, shutil, random
  
base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val" 
train_dir = base_dir / "train" 
for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname,
                    val_dir / category / fname)
```

Then, create three `Dataset` objects for training, validation, and testing using the `keras.utils.text_dataset_from_directory` from the [`Keras API`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory):

```python
from tensorflow import keras
batch_size = 32 
  
train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)
```

These datasets contain inputs that are TensorFlow `tf.string` `int32` tensors and targets encoding the value `0` or `1`. 

Compared to one-hot vectors that are typically sparse, high-dimensional, and inefficient to use, word embeddings significantly improve the overall quality of the word representations and, consequently, the learning algorithm. 

So, in this step, you'll download and use the GloVe word embeddings, then parse them, load the word vectors into a [`Keras Embedding`](https://keras.io/api/layers/core_layers/embedding/) layer, and then finally build a new model that uses it.


### Step 3: Download pretrained GloVe embeddings

```bash
!wget http:/ /nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip
```


### Step 4: Parse the GloVe word-embeddings file

```python
import numpy as np
path_to_glove_file = "glove.6B.100d.txt"

embeddings_index = {} 
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")

        # Builds an index that maps words (as strings) to their vector representations
        embeddings_index[word] = coefs
```


### Step 5: Prepare the GloVe word-embeddings matrix

```python
from tensorflow.keras import layers
max_length = 600
max_tokens = 20000
embedding_dim = 100 

# Here, the input reviews are truncated after 600 words since only 5% reviews are longer than 600 words
text_vectorization = layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=max_length)

# Get the vocabulary indexed by the TextVectorization layer
vocabulary = text_vectorization.get_vocabulary()

# Create a mapping from words to their index in the vocabulary
word_index = dict(zip(vocabulary, range(len(vocabulary))))

# Matrix that you'll fill GloVe vectors in
embedding_matrix = np.zeros((max_tokens, embedding_dim))
for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```


### Step 6: Use positional embeddings

Unlike word embeddings, positional encodings give the model access to word-order information by adding to each word embedding the word's position in the sentence. This causes the input word embeddings to have two components: the usual word vector that represents the word independent of any context, and the position vector, that represents the position of the word in the current sentence.

You'll add the position embeddings to the corresponding word embeddings, to obtain a position-aware word embeddings, also called positional embeddings.

```python
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):  
        super().__init__(**kwargs)

        # Prepare an embedding layer for the token indices
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
  
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)

        # Add both embedding vectors together
        return embedded_tokens + embedded_positions

    # Masking takes care of zeros padded to inputs in a batch
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    # Serialization so we can save the model
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
```


### Step 7: Run the Transformer

Finally, run the Transformer to check how well it correctly classified the movie reviews based on their sentiment:

```python
vocab_size = 20000 
sequence_length = 600 
embed_dim = 256 
num_heads = 2 
dense_dim = 32 

inputs = keras.Input(shape=(None,), dtype="int64")

# Use the positional embedding layer like a regular embedding layer
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("full_transformer_encoder.keras",
                                    save_best_only=True)
] 
model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)
model = keras.models.load_model(
    "full_transformer_encoder.keras",
    custom_objects={"TransformerEncoder": TransformerEncoder,
                    "PositionalEmbedding": PositionalEmbedding}) 
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f} %")
```

If everything goes right, you should get a test accuracy that's similar to the following output:

> Test acc: 88.3%


Check out [How Do Transformers Work?](important-concepts.md) to learn how the Transformer model works.

Furthermore, see [How To Use Transformers for Machine Translation](how-to-use-transformers-for-translation.md) to learn how you can use Transformers for machine translation.