# How To Use Transformers for Machine Translation

In this tutorial, you'll work on a machine translation problem of translating English sentences into Spanish. The dataset you'll use for this task is the English-Spanish sentence pairs from the [Tatoeba Project](https://www.manythings.org/anki/).


## Steps to perfom machine translation

### Step 1: Download the dataset

```bash
!wget http:/ /storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
!unzip -q spa-eng.zip
```

### Step 2: Prepare the dataset

The text file contains one example per line: an English sentence, followed by a tab character, followed by the corresponding Spanish sentence. So, you'll have to parse the file:

```python
text_file = "spa-eng/spa.txt" 
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = [] 
for line in lines:
    english, spanish = line.split("\t")
    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))
```

After this step, the sentence pairs look as follows:

```bash
>>> import random
>>> print(random.choice(text_pairs))
("Soccer is more popular than tennis.",
 "[start] El fútbol es más popular que el tenis. [end]")
 ```

Then, shuffle and split them into the usual training, validation, and test sets:

```python
import random
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]
```

### Step 3: Vectorise the English-Spanish text pairs

Prepare two separate `TextVectorization` layers: one for English and one for Spanish.

```python
import tensorflow as tf 
import string
import re
from tensorflow.keras import layers

batch_size = 64 
  
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
 
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")
vocab_size = 15000
sequence_length = 20


def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({
        "english": eng,
        "spanish": spa[:, :-1],
    }, spa[:, 1:])
 

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()
 
source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
```


### Step 4: Download pretrained GloVe embeddings

Compared to one-hot vectors that are typically sparse, high-dimensional, and inefficient to use, word embeddings significantly improve the overall quality of the word representations and, consequently, the learning algorithm. 

So, in the next few steps, you'll download and use the GloVe word embeddings, then parse them, load the word vectors into a [`Keras Embedding`](https://keras.io/api/layers/core_layers/embedding/) layer, and then finally build a new model that uses it.

```bash
!wget http:/ /nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip
```


### Step 5: Parse the GloVe word-embeddings file

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


### Step 6: Prepare the GloVe word-embeddings matrix

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


### Step 7: Use positional embeddings

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

### Step 8: Design the Transformer encoder

```python
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        
        # size of the input token vectors
        self.embed_dim = embed_dim
        # size of the inner dense layer
        self.dense_dim = dense_dim
        # number of attention heads
        self.num_heads = num_heads
        
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(emed_dim), ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
    
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis:1]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs+attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input+proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
            "dense_dim": self.dense_dim
            "num_heads": self.num_heads
        })
        return config
```


### Step 9: Design the Transformer decoder

The `TransformerDecoder` is similar to the `TransformerEncoder`, except it features an additional attention block where the keys and values are the source sequence encoded by the `TransformerEncoder`, as shown below. Together, the encoder and the decoder form an end-to-end Transformer.


**Prepare the `TransformerDecoder` class**

```python
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True                     
  
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
```


**Causal Padding**

_Causal padding_ is absolutely critical to successfully training a seq2seq Transformer. Unlike an RNN, the `TransformerDecoder` is order-agnostic, i.e., it looks at the entire target sequence at once. 

For causal padding, mask the upper half of the pairwise attention matrix to prevent the model from paying any attention to information from the future: only information from tokens `0...N` in the target sequence should be used when generating target token N+1. This is done in the `get_causal_attention_mask` method within the `TransformerDecoder` to retrieve an attention mask that can be passed to the `MultiHeadAttention` layers.

```python
def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")                           
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))    
        mult = tf.concat(                                               
            [tf.expand_dims(batch_size, -1),                            
             tf.constant([1, 1], dtype=tf.int32)], axis=0)             
        return tf.tile(mask, mult)
```


**Forward-pass of the `TransformerDecoder` class**

```python
def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)       
        if mask is not None:                                       
            padding_mask = tf.cast(                                
                mask[:, tf.newaxis, :], dtype="int32")             
            padding_mask = tf.minimum(padding_mask, causal_mask)   
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)                            
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,                           
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)
```


### Step 10: Run the Transformer

Finally, it's time to perform some translation.

```python
embed_dim = 256 
dense_dim = 2048 
num_heads = 8 
  
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)  
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)        
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
transformer.fit(train_ds, epochs=30, validation_data=val_ds)
```

### Step 11: Translate new sentences

Now, try using the model you just trained to translate never-seen-before English sentences from the test set.

```python
import numpy as np
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20 
  
def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]" 
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])  
        sampled_token_index = np.argmax(predictions[0, i, :])       
        sampled_token = spa_index_lookup[sampled_token_index]       
        decoded_sentence += " " + sampled_token                     
        if sampled_token == "[end]":                                
            break                                                   
    return decoded_sentence
  
test_eng_texts = [pair[0] for pair in test_pairs] 
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
```

After the run is complete, sample some results from the Transformer translation model, which could be similar to:

> This is a song I learned when I was a kid.  
> [start] esta es una canción que aprendí cuando era chico [end]    
>
> She can play the piano.  
> [start] ella puede tocar piano [end]
>
>I'm not who you think I am.  
>[start] no soy la persona que tú creo que soy [end]
>
>It may have rained a little last night.  
>[start] puede que llueve un poco el pasado [end]


> **IMPORTANT**
>* Translation models often make unwarranted assumptions about their input data, which leads to ***algorithmic bias***. 
>* In the worst cases, the model might hallucinate memorized information that has nothing to do with the data it’s currently processing.


Check out the [How Do Transformers Work?](Important%20Concepts.md) section if you are still curious about Transformers and would like to learn more about these models.