# Medical NLP Question Answering 
*Developed between April and May 2025 at Politecnico di Milano*

The aim of the project was to study the **Medical Meadow Medical Flashcards** dataset and develop multiple **Question Answering LLM** and compair their performances. Lastly a **voice interactive system** version was
implemented.

## Dataset: Medical Meadow Medical Flashcards

* Website: https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards
* Paper: https://arxiv.org/pdf/2304.08247.pdf
* Description: Information on medical curriculum flashcards has been given to GPT-3.5 and used to
create medical knowledge question answer pairs.
* Task: Medical Question Answering (i.e. train a model to answer medical questions.)

## Summary 

- Preliminary analysis
  - Data Exploration and Visualization
  - Word Embedding
  - Clustering
  - Context Retrival: Using PubMedAPI we extract the *passage* where answer can be found (TF-IDF and BM25 Index Search)
- Training models
  - Pretrained models fine-tuned on our dataset: well know models fine-tuned on out task and datset
    - **TinyLlama**
    - **Gemma (7B)**
    - **MISTRAL (7B)**
    - **Llama 3 (8B)**
    - **GPT-2**
 -  Pretrained context-based model: The context retrieved before is used to train a BERT based model in a supervised way using (question, context) -> answer pairs
   - **BioBert**
- Testing and performance scores
  - **BertScore**
  - Evaluation with Embedding: **BioSentVec**
- Possible extensions: **Voice Interactive System**

