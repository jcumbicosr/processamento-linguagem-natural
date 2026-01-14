# NLP and LLM Implementation Experiments

This repository contains a collection of Jupyter Notebooks demonstrating various techniques in Natural Language Processing (NLP) and Large Language Models (LLMs). The projects range from classical Neural Language Models to modern Transformer architectures, Fine-tuning techniques (LoRA), and Multimodal Search (CLIP-style).

Most language models in this repository are trained using the **Machado de Assis** dataset.

## 📂 Repository Contents

### 1. Language Modeling Architectures

* **`Modelo_de_Linguagem_(Bengio_2003)_MLP_+_Embeddings.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcumbicosr/processamento-linguagem-natural/blob/main/Modelo_de_Linguagem_(Bengio_2003)_MLP_%2B_Embeddings.ipynb)

    * **Description:** Implementation of the classic Neural Probabilistic Language Model by Bengio et al. (2003).
    * **Implementation Details:** This notebook constructs a feed-forward neural network (MLP) that learns a distributed representation for words. It uses a fixed-size context window (n-gram approach) where input words are mapped to dense vectors via an **Embedding layer**. These vectors are concatenated and passed through linear hidden layers with non-linear activations (tanh/ReLU) to predict the probability distribution of the next word via a **Softmax** output layer.
    * **Highlights:** Demonstrates the fundamental shift from count-based n-grams to distributed representations, allowing the model to generalize better to unseen contexts by leveraging word similarity in the embedding space.

* **`Modelo de Linguagem com auto atenção.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcumbicosr/processamento-linguagem-natural/blob/main/Modelo_de_Linguagem_com_auto_atenção.ipynb)
    * **Description:** A language model implemented using **Self-Attention** mechanisms.
    * **Implementation Details:** This model replaces the fixed context window of the MLP with a **Self-Attention** mechanism. It computes **Query, Key, and Value** matrices to determine the relevance of each word in the input sequence relative to others. The implementation focuses on the matrix operations required to calculate attention scores, normalize them via Softmax, and compute the weighted sum of values.
    * **Highlights:** Moves beyond the fixed-window limitation of Bengio's model, allowing the network to dynamically weigh the importance of different words in the context regardless of their distance from the target word.

* **`Modelo de Linguagem com auto atenção e máscara causal.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcumbicosr/processamento-linguagem-natural/blob/main/Modelo_de_Linguagem_com_auto_atenção_e_máscara_causal.ipynb)
    * **Description:** Implementation of a GPT-style language model using **Causal Self-Attention**.
    * **Implementation Details:** This notebook refines the self-attention mechanism by adding a **Causal Mask** (an upper triangular matrix of negative infinity) to the attention scores before Softmax. This ensures the model preserves the autoregressive property, preventing it from attending to future tokens during training. The pipeline includes a custom tokenizer handling `<sos>` (Start of Sequence) and `<eos>` (End of Sequence) tokens, and a dataset loader that shifts targets one step to the right.
    * **Highlights:** Successfully implements the core architecture behind decoder-only transformers like GPT, enabling coherent text generation by predicting the next token based strictly on past context.

### 2. Embeddings & Pre-trained Models

* **`Embeddings gerados por BERT pré-treinado`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcumbicosr/processamento-linguagem-natural/blob/main/Embeddings_gerados_por_BERT_pré_treinado.ipynb)
    * **Description:** Exploration of using a pre-trained **BERT** model to generate contextualized word embeddings.
    * **Implementation Details:** The notebook leverages the **Hugging Face Transformers** library to load a pre-trained BERT model. It demonstrates how to tokenize input text, pass it through the model, and extract hidden states from the final layers. Unlike static embeddings (Word2Vec), these embeddings are dynamic, changing based on the sentence context.
    * **Highlights:** Visualizes how BERT handles polysemy (words with multiple meanings) by generating different embedding vectors for the same word depending on its surrounding context.

### 3. Fine-Tuning Techniques

* **`Ajuste fino usando LoRA.ipynb`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcumbicosr/processamento-linguagem-natural/blob/main/Ajuste_fino_usando_LoRA.ipynb)
    * **Description:** Implementation of **LoRA (Low-Rank Adaptation)** to fine-tune the Causal Self-Attention model.
    * **Implementation Details:** The model is first pre-trained on a 70% split of the Machado de Assis dataset. For fine-tuning, the original weights are frozen, and low-rank matrices **A and B** are injected into the linear layers (specifically targeting query/value projections or embeddings). The model is then trained on the remaining 20% of data, updating only these low-rank matrices.
    * **Highlights:** Demonstrates **Parameter-Efficient Fine-Tuning (PEFT)**, significantly reducing the computational cost and memory footprint of training while effectively adapting the model to new data distributions using only a fraction of the total parameters.

### 4. Multimodal Learning

* **`Busca Multimodal por Embedding`** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcumbicosr/processamento-linguagem-natural/blob/main/Busca_Multimodal_por_Embedding.ipynb)
    * **Description:** A Multimodal search system aligning Image and Text embeddings (similar to **CLIP**).
    * **Implementation Details:** This project uses separate, frozen pre-trained encoders for images and text. It learns linear projections to map the output of both encoders into a shared embedding space. The training process initially uses **MSE Loss** to align positive pairs and evolves to implement **Contrastive Loss**, maximizing the cosine similarity between correct image-text pairs while minimizing it for incorrect ones within a batch.
    * **Highlights:** Results in a functional **Text-to-Image search engine**, allowing users to query the image database using natural language descriptions. The shift to Contrastive Loss notably improves the separation between dissimilar concepts compared to simple MSE alignment.

## 🛠️ Technologies Used

* **Python**
* **PyTorch** (Deep Learning Framework)
* **Hugging Face Transformers** (for BERT/Pre-trained models)
* **Jupyter Notebooks**
* **Matplotlib** (Visualization)

## 🚀 How to Run

1.  Clone this repository.
2.  Ensure you have Python and Jupyter installed.
3.  Install the dependencies (mainly PyTorch and standard scientific libraries).
4.  Open the notebooks in Jupyter Lab, Jupyter Notebook, or Google Colab.
    * *Note: Some notebooks may require GPU acceleration for faster training.*

---
*Created for educational purposes in the context of Deep Learning and NLP studies.*
