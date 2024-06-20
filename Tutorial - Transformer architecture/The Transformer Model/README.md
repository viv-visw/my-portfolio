# The Transformer


The [Transformer model](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) is a popular deep learning architecture that was introduced in 2017 by researchers at Google Brain and the University of Toronto, Canada.

Based on assigning weighted importance to different parts of a sentence to improve its long-range temporal context, the Transformer model has been applied to a wide variety of tasks, from natural language to computer vision, finance, and even cybersecurity.


## Overview

Conceptually, the original Transformers architecture is based on the idea of mimicking the way humans selectively focus only on specific parts of an input sentence. This specific focus enhances its long-range context and memory both during the encoder input and decoder output stages, unlike recurrent and LSTM recurrent networks that suffer from a relatively shorter context memory, resulting in an inferior semantic quality of model output. Moreover, this weighted-attention design has become the foundation of a more sophisticated class of models called large language models that have redefined the state-of-the-art across several natural language processing tasks, including text prediction, text summarization, question-answering, chatbot applications, and much more.

This project helps you:

* Understand the basics of the Transformer model.

* Learn how the Transformer model works.

* Gain an intuition about all the important details you'd need.

* Use the model for a real-world language translation task.


## Audience

This documentation is meant for you, if you're trying to understand:

* how Transformer models work.

* the underlying self-attention mechanism.

* how you can implement Transformer models.

* how you can run a language translation task using the Transformer model.


## Prerequisites

This documentation assumes you understand:

* foundational concepts in machine learning and deep learning, especially:

    * linear algebra

    * differential calculus

* how to write software code in Python and use deep learning libraries such as Keras and TensorFlow.

* how recurrent neural networks, [sequence-to-sequence models](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) that has an encoder-decoder design, and the attention mechanism work.

* the basics of natural language processing like [vectorisation, tokenisation, and embeddings](https://web.stanford.edu/class/cs224n/).


## Scope

This project aims to answer the following questions about the Transformer model:

* What is the Transformer model?

* Why is the Transformer model more successful than traditional sequence-to-sequence (seq2seq) deep learning models?

* How does the theoretical deisgn of the Transformer model ensure improved learning, in comparison to traditional deep recurrent neural networks?

* How can I use the Transformer model for language translation?


## Table of Contents

| Topics | Links |
| :------:| :-----: |
| Prerequisites | [Prerequisites](prerequisites.md)
| Getting Started with Transformers| [Getting Started with Transformers](getting-started.md)
| How To Implement Transformers for Machine Translation | [Transformers for Machine Translation](how-to-use-transformers-for-translation.md)
| Theory of Transformers| [Transformer Fundamentals](important-concepts.md)


## Author

**Vivek Viswanath**
