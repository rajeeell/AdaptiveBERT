# AdaptiveBERT |  Domain-Specific Fine-Tuning for Text Classification

This repository demonstrates the fine-tuning of Google's DistilBERT model for classifying news articles into categories such as **tech, business, politics, entertainment, and sports**. It includes **data preprocessing scripts (learning.py), model training, and prediction scripts (predict.py)**. Due to size constraints, the fine-tuned model itself is not included in the repository.


## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Data Preprocessing](#data-preprocessing)
- [Usage](#usage)
- [Model Fine-Tuning](#model-fine-tuning)

## Introduction

AdaptiveBERT is a project that harnesses cutting-edge natural language processing (NLP) techniques to categorize news articles into predefined topics. This project relies on [Hugging Face Transformers](https://huggingface.co/transformers/) and [TensorFlow](https://www.tensorflow.org/) for its implementation.

## Achieved Results

1. Developed an NLP pipeline to train Google’s DistilBERT model for multi-class text classification using TensorFlow and Hugging Face Transformers.  
2. Fine-tuned the model on a custom BBC text classification dataset to enhance domain-specific accuracy.  
3. Leveraged DistilBERT’s efficiency, achieving 97% of BERT’s language understanding while reducing model size by 40% and accelerating training by 60%.


## Getting Started

To begin using this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/rajeeell/AdaptiveBERT.git
Install the required dependencies:
   ```bash
   pip install transformers tensorflow pandas scikit-learn
   ```

## Data Preprocessing
Before using the model, ensure your dataset is preprocessed. If you are working with a custom dataset, follow these steps:  

1. Tokenize the dataset and assign labels to each news article.
     
2. Convert the labels into a binary format to align with the model's requirements.
   
3. For a quick start, use the provided example dataset (BBC Text Classification) and follow the preprocessing steps outlined in the code.
   
4. Fine-tune the DistilBERT model on the preprocessed dataset.
   
5. Save the fine-tuned model in the **saved_models** directory, ensuring you have the necessary write permissions for your operating system.

   
## Usage
You can use the fine-tuned DistilBERT model to classify news articles. Run the following script and enter the news article text when prompted:

bash
Copy code
python predict.py
The script will categorize the news article into one of the following classifications based on the provided input: Business, Entertainment, Politics, Sport, or Tech.

## Model Fine-Tuning
To fine-tune the DistilBERT model on your own dataset, follow these steps:  

1. Format your dataset similarly to the provided BBC Text Classification dataset.  

2. Update the code to load and preprocess your dataset accordingly.  

3. Use the TFTrainer to fine-tune the model.  

4. Save the fine-tuned model in the **saved_models** directory.

## Contributing
Contributions to this project are welcome! If you encounter issues or have suggestions for improvements, please open an issue or create a pull request.
