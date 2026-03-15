# Document Classification using Longformer

Deep learning system for **classifying long scientific documents** using
the Longformer transformer architecture.\
This project demonstrates how modern transformer architectures can
efficiently process **long sequences** and improve classification
performance on research papers.

------------------------------------------------------------------------

## Project Badges

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Dataset](https://img.shields.io/badge/Dataset-arXiv-red)
![License](https://img.shields.io/badge/License-MIT-green)

------------------------------------------------------------------------

## Overview

Document classification is an important task in **Natural Language
Processing (NLP)** where textual documents are automatically assigned to
predefined categories.

With the exponential growth of digital documents in domains such as:

-   scientific publications
-   healthcare documentation
-   legal records
-   news articles

automatic classification systems are required to efficiently organize
and retrieve information.

Traditional transformer models struggle with long documents because of
sequence length limitations.\
This project uses **Longformer**, which introduces **sparse attention
mechanisms** to process long text sequences efficiently.

------------------------------------------------------------------------

## Dataset

The dataset used in this project is derived from the **arXiv research
paper dataset**.

Dataset Source:\
https://www.kaggle.com/datasets/Cornell-University/arxiv

### Dataset Features

  Feature    Description
  ---------- ----------------------
  Title      Research paper title
  Abstract   Summary of the paper
  Category   Research domain

Model Input:

    Title + Abstract

------------------------------------------------------------------------

## Data Preprocessing

![Data Preprocessing](images/data_preprocessing.png)

### Preprocessing Steps

1.  Remove missing values
2.  Remove unnecessary metadata
3.  Convert text to lowercase
4.  Remove special characters
5.  Combine title and abstract
6.  Prepare data for tokenization

------------------------------------------------------------------------

## Longformer Architecture

Longformer is designed to process **long documents efficiently** using
sparse attention.

### Key Features

-   Supports sequences up to **4096 tokens**
-   Uses **sliding window attention**
-   Reduces computational complexity
-   Maintains contextual understanding of long texts

------------------------------------------------------------------------

## Sliding Window Attention

![Sliding Window Attention](images/sliding_window.png)

Traditional transformer complexity:

    O(n²)

Longformer complexity:

    O(n × w)

Where:

-   **n** = sequence length\
-   **w** = attention window size

------------------------------------------------------------------------

## Training Configuration

  Parameter       Value
  --------------- ---------------------
  Model           Longformer
  Tokenizer       LongformerTokenizer
  Optimizer       AdamW
  Loss Function   CrossEntropyLoss
  Learning Rate   5e-5
  Epochs          3
  Batch Size      4

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/Er-Robinson/Document-Classification-using-Longformer.git
cd Document-Classification-using-Longformer
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Required libraries:

-   PyTorch
-   Transformers
-   NumPy
-   Pandas
-   Scikit-learn

------------------------------------------------------------------------

## Model Training Example

``` python
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch

model_name = "allenai/longformer-base-4096"

tokenizer = LongformerTokenizer.from_pretrained(model_name)

model = LongformerForSequenceClassification.from_pretrained(
    model_name,
    num_labels=10
)

inputs = tokenizer(
    "Sample research paper abstract text",
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=4096
)

outputs = model(**inputs)

logits = outputs.logits
prediction = torch.argmax(logits)

print("Predicted class:", prediction)
```

------------------------------------------------------------------------

## Experimental Results

### BERT Baseline

![BERT Accuracy](results/bert_accuracy.jpg)

BERT struggles with long documents due to the **512 token limit**.

------------------------------------------------------------------------

### Longformer Base Model

![Longformer Accuracy](results/long_accuracy.png)

Longformer captures long-range dependencies and improves classification
performance.

------------------------------------------------------------------------

### Longformer Large Model

![Longformer Large Accuracy](results/longLarge_accuracy.png)

The larger Longformer architecture provides improved representation
learning.

------------------------------------------------------------------------

### Additional Longformer Evaluation

![Longformer Large Accuracy 2](results/longLarge_accuracy2.png)

------------------------------------------------------------------------

## Overall Model Accuracy

![Overall Accuracy](results/overall_acc.png)

**Final Accuracy: 84.3%**

------------------------------------------------------------------------

## Prediction Examples

### Prediction Example 1

![Prediction](results/prediction.png)

### Prediction Example 2

![Prediction](results/prediction2.png)

------------------------------------------------------------------------

## Project Structure

    Document-Classification-using-Longformer
    │
    ├── data
    ├── images
    │   ├── data_preprocessing.png
    │   └── sliding_window.png
    ├── results
    │   ├── bert_accuracy.jpg
    │   ├── long_accuracy.png
    │   ├── longLarge_accuracy.png
    │   ├── longLarge_accuracy2.png
    │   ├── overall_acc.png
    │   ├── prediction.png
    │   └── prediction2.png
    ├── models
    ├── notebooks
    ├── src
    └── README.md

------------------------------------------------------------------------

## Applications

This system can be used for:

-   scientific literature classification
-   legal document categorization
-   news article classification
-   academic search engines
-   digital library indexing

------------------------------------------------------------------------

## Future Work

Possible improvements:

-   Training on **full research papers instead of abstracts**
-   Exploring models such as **BigBird**
-   Adding **attention visualization**
-   Deploying the model as an **API service**

------------------------------------------------------------------------

## References

-   Beltagy, I., Peters, M., & Cohan, A. (2020) -- *Longformer: The
    Long-Document Transformer*
-   Devlin, J. et al. (2019) -- *BERT: Pre-training of Deep
    Bidirectional Transformers*
-   Vaswani, A. et al. (2017) -- *Attention Is All You Need*

------------------------------------------------------------------------

## Author

**Robinson**

Machine Learning • Natural Language Processing • Deep Learning

GitHub: https://github.com/Er-Robinson\
LinkedIn: https://linkedin.com/in/robinson-189312161

------------------------------------------------------------------------

⭐ If you find this project useful, please consider **starring the
repository**.
