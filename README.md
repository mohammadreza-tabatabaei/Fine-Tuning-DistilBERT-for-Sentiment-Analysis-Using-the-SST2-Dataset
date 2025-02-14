# Fine-Tuning DistilBERT for Sentiment Analysis Using the SST2 Dataset

This project demonstrates how to fine-tune the pre-trained **DistilBERT** model for sentiment analysis on the **SST2** dataset. The model is fine-tuned using the **TensorFlow** framework, and the goal is to classify sentences into positive or negative sentiments.

## Project Overview

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment expressed in a piece of text. The **SST2 dataset** (Stanford Sentiment Treebank 2) consists of movie reviews labeled as either positive or negative. 

In this project, we use the **DistilBERT** model, a smaller and faster version of BERT, to perform sentiment classification on the SST2 dataset. The model is fine-tuned on the dataset, and the results are evaluated using accuracy and loss metrics.

## Steps

### Dataset Loading
The SST2 dataset is loaded using the `datasets` library from Hugging Face.

### Tokenization
The dataset is tokenized using the **DistilBERT** tokenizer, converting text into numerical format for the model.

### Model Setup
The pre-trained **DistilBERT** model is loaded and configured for sequence classification.

### Model Fine-Tuning
The model is fine-tuned on the SST2 dataset by training the final classification layer while freezing the other layers.

### Training and Evaluation
The model is trained on the training set and validated on the validation set. The performance is evaluated using accuracy and loss.

### Saving the Model
After training, the fine-tuned model is saved for future use.

## Results
After training the model for 3 epochs, the performance is evaluated on the validation set. The metrics include:

- **Test Loss**
- **Test Accuracy**

## Usage
You can use the fine-tuned model for sentiment analysis by loading it and passing input text through the tokenizer. The model will predict whether the sentiment is positive or negative.

## License
This project is licensed under the MIT License - see the LICENSE file for details.



