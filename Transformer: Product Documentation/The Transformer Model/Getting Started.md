# **Transformers**: Getting Started


## Overview

This article provides a quick walkthrough of the steps you can complete, as you read along, to use the Transformer model and see it in action! To gain a deeper understanding of how Transformers work, see the [How Do Transformers Work?](/Transformer%3A%20Product%20Documentation/The%20Transformer%20Model/Important%20Concepts.md) section.

As an illustration, you will work on a text sentiment-classification problem as a Natural Language Processing (NLP) task. The dataset you will use for this task is the [IMDB Movie Review Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).



## Prerequisites

* You have prior experience with Python programming.
* You already have deep learning libraries such as [`Keras`](https://keras.io/getting_started/) and [`TensorFlow`](https://www.tensorflow.org/install) installed.
* You already have an understanding of NLP topics such as [vectorisation, tokenisation, and embeddings](https://web.stanford.edu/class/cs224n/).
* You already understand the basics of traditional [sequence-to-sequence models](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) with encoder-decoder components.



## Procedure

> ### Step 1: Download the dataset

```bash
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```
  
You’re left with a directory named aclImdb, with the following structure:

```
aclImdb/
...train/
......pos/
......neg/
...test/
......pos/
......neg/
```

For instance, the `train/pos/` directory contains a set of 12,500 text files, each of which contains the text body of a positive-sentiment movie review to be used as training data. The negative-sentiment reviews live in the “neg” directories. In total, there are 25,000 text files for training and another 25,000 for testing.

There’s also a `train/unsup` subdirectory in there, which you don’t need. So, delete it by running:

```bash
!rm -r aclImdb/train/unsup
```

> ### Step 2: Prepare the dataset

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

Then, create three Dataset objects for training, validation, and testing using the `keras.utils.text_dataset_from_directory` from the [`Keras API`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory):

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

> ### Step 3: Prepare integer sequence datasets

In this step, you prepare datasets that return integer sequences.

```python
from tensorflow.keras import layers
  
max_length = 600 
max_tokens = 20000 
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,     
)
text_vectorization.adapt(text_only_train_ds)
 
int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
```

 > ✅ **NOTE**
 > Here, the inputs are truncated after the first 600 words to keep a manageable input size. Also, vectors are 20000-dimensional vectors.


> ### Step 4: Use Pretrained Word Embeddings

As you would know, word embeddings significantly improve the overall quality of thee learning aglorithm. So, in this step, you will start by downloading the GloVe files, then parse them, load the word vectors into a [`Keras Embedding`](https://keras.io/api/layers/core_layers/embedding/) layer, and then finally build a new model that uses it.

**Download GloVe embeddings**

```bash
!wget http:/ /nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip
```
**Parse the GloVe word-embeddings file**

```python
import numpy as np
path_to_glove_file = "glove.6B.100d.txt" 
  
embeddings_index = {} 
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
```

**Prepare the GloVe word-embeddings matrix**

```python
embedding_dim = 100 
  
vocabulary = text_vectorization.get_vocabulary()             
word_index = dict(zip(vocabulary, range(len(vocabulary))))   
 
embedding_matrix = np.zeros((max_tokens, embedding_dim))     
for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:                         
        embedding_matrix[i] = embedding_vector    
```


> ### Step 5: Use Positional Embeddings as a subclassed layer

Unlike embeddings, positional encodings give the model access to word-order information by adding the word's position in the sentence to each word embedding.

Based on the original ["Attention is all you need" Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) paper, you will proceed to add the position embeddings to the corresponding word embeddings, to obtain a position-aware word embedding. This technique is called “positional embedding".

```python
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):  
        super().__init__(**kwargs)
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
        return embedded_tokens + embedded_positions                        
 
    def compute_mask(self, inputs, mask=None):                             
        return tf.math.not_equal(inputs, 0)                                
 
    def get_config(self):                                                  
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
```


> ### Step 6: Run the Transformer as a text classifier

Finally, it's time to run the Transformer!

```python
vocab_size = 20000 
sequence_length = 600 
embed_dim = 256 
num_heads = 2 
dense_dim = 32 
  
inputs = keras.Input(shape=(None,), dtype="int64")
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

> ✅ **NOTE**  
> Check the `keras.layers.Layer` API for the `TransformerEncoder` class.


## Result

If everything goes right (fingers crossed!), then you should an output with very similar performance to the following output:

> Test acc: 88.3%


## Further Reading

Congratulations on running your first Transformer model!  

If you would like to learn more about Transformers, check out the [How Do Transformers Work?](/Transformer%3A%20Product%20Documentation/The%20Transformer%20Model/Important%20Concepts.md) section.

If you already understand the basics and just want to see another scenario with Transformers in action, see [How To Use Transformers for Machine Translation](/Transformer%3A%20Product%20Documentation/The%20Transformer%20Model/How%20to%20use%20Transformers%20for%20Translation.md)!