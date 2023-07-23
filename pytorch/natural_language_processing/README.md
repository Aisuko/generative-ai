
## Natural Language Processing(NLP) with PyTorch

We will explore different neural network architectures for dealing with natural language texts. And more:
* using bag-of-words(BoW)- classical NLP architectures
* word embeddings
* recurrent neural networks(RNNs) for text classification from news headlines to one of the 4 categories(World, Sports, Business and SciTech)
* generative neural networks(GNNs)
* build text classification models


## Introduction

NLP has experienced fast growth primarily due to the performance of the language models' ability to accurately "understand" human language faster while using unsupervised training on large text corpora. For instance, sentence generation using GPT-3 or ppre-trained text model such as BERT.

## Notebooks
|No|Title|Kaggle|
|---|---|---|
|1|[Representing text as Tensors](representing_text_as_tensors.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/aisuko/representing-text-as-tensors)|
|2|[Represent words with embeddings](represent_words_with_embeddings.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/aisuko/represent-word-with-embeddings)|
|3|[Capture patterns with RNN(working on)](capture_patterns_with_recurrent_neural_networks.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/aisuko/)|

## Netural Language Tasks

There are several NLP tasks that we traditionally try to solve using neural networks:

* **Text Classification** is used when we need to classify text fragment into one of several pre-defined classes. Examples include e-mail spam detection, news categotizationm, assigning support request to one of the categories, and more.
* **Intent Classification** is one specific case of text classification, when we want to map input utterance in the conversational AI system into one of the intents that represent the actual meaning of the phrase, or intent of the user.
* **Sentiment Nalysis** is a regression task, where we want to understand the degree of negativity of given piece of text. We may want to label texts in a dataset from the most negative (-1) to most postive ones (+1), and train a model taht will output a number of "positiveness" of a text.
* **Named Entity Recognition (NER)** is a task of extracing some entities from text, such as dates, addresses, people names, etc. Together with intent classification, NER is often used in dialog systems to extract parameters from user's utterance.
* A similar task of **keyword extraction** can be used to find the most meaningful words inside a text, which can then be used as tags.
* **Text Summarization** extracts the most meaningful pieces of text, giving a user a compressed version taht contains most of the meaning.
* **Question/Answer** is a task of extracting an answer from a piece of text. This model gets text fragment and a question as an input, and needs to find exact place within the text that contains answer.

Here we will mostly focus on the **text classification** taks. We'll use text from news headlines to classify which onw of the 4 categories they belongs to:
* World Sports
* Business and Sci/Tech

We also introduce generative models that can self produce human-like text sequences.
