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

**note:** *check the notebooks for more datails*

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
- Extensions: **Voice Interactive System**

Complete analysis: 

Here [general report](https://github.com/lorenzofranze/NLP-medical-Q-A/blob/main/Report.ipynb) is shown for convencience along with conclusions but have a look at each notebook individually for more details
# Report

## 1. Preliminary analysis:

### Data Exploration and Visualization

Our work began with a set of preliminary analyses on the dataset. These analyses provided valuable insights that are crucial for developing our question-answering model in the best way possible. The dataset will be cleaned, and the following analyses and operation will be performed:


*   Removal of invalid and empty entries to ensure data quality and consistency

*   Comprehensive preprocessing operations, including the removal of punctuation, stopwords, and lemmatization of the questions and of the answers, to stndardize the text and improve model performance. Every preprocessing step separately to allow for later comparisons, such as analyzing senetece length with and without stopwords or lematization

*  Examination of word counts per question and answer to understand length and complexity, vocabulary analysis for both questions and answers, revealing the diversity and richness of language usage, and distribution analysis of words across the dataset, providing insights into frequency and spread

*  WordCloud analysis and identification of the 100 most common words to visualize and highlight the most significant terms within the dataset

*  Visualization of N-grams related to the question and answer sets to uncover common phrases, patterns, and linguistic structures




### Clustering


In this phase, we conducted a comprehensive analysis using clustering methodologies of a medical dataset aimed at developing a question-answering model. Our analysis involved utilizing preprocessed data and conducting clustering and visualization techniques to gain insights into the structure and content of the dataset.



*   **Data Exploration and Preprocessing:**
We began by exploring the dataset's characteristics, including word counts, vocabulary richness, and distribution of words. Preprocessing steps such as removing punctuation, stopwords, and lemmatization were applied to standardize the text and improve model performance.
*   **Visualization and Analysis:**
We visualized the dataset using techniques like WordClouds, N-grams, and TF-IDF analysis to uncover common phrases, patterns, and significant terms. This allowed us to gain a deeper understanding of the dataset's content and identify prevalent medical themes.
*   **Clustering Analysis:**
Utilizing MiniBatch K-Means clustering, we identified clusters of documents based on their TF-IDF vectorized terms. The clustering revealed dominant topics within the dataset, such as vitamin deficiency, cancer types, thyroid conditions, and patient-related discussions.
*   **Evaluation and Insights:**
Evaluation metrics such as inertia and silhouette coefficient were used to assess the clustering performance. Despite achieving meaningful clusters, the silhouette coefficient indicated room for improvement, suggesting potential overlap between clusters and the need for further refinement.


### Word Embedding

Another crucial step involves performinng word and sentence embedding on our dataset to capture semantic meaning and relationships between words based on their usage in large text corpora. We explore two methods of working with embeddings: Learning word embeddings representation and Loading pre-trained models for word embedding.



1.   **Learning Word Embedding**:  Our journey begins with learning the embedding representation of our dataset. The questions and answers undergo a preprocessing process, including the removal of punctuation and special characters, expanding contractions, converting words to lowercase, and lemmatization.
We empoly the Word2Vec model to obtain word representations trained on our dataset. Subsequently, we examine embeddings for a words within our vocabulary and identify similar words. Finally, we visualize the embedding vectors using **t-SNE**



2.   **Loading Pre-trained models**: To obtain better performances, we leverage pre-trained models for word embeddings. These models, trained on large corpora of data caputre richer semantic meanings and relationships between words. We experimented with different models, such as **Word2Vec**, **Glove** and **fastText**, pre-trained on datasets both specific to our medical domain and more general.
We performed the same operations as in the case of learning word embeddings. Noteworthy is fastText, as it enables us to compute embeddings for out-of-vocabulary words and perform sentence embeddings, a technique we will explore further in subsequent notebooks.


The analysis clearly demonstrates that utilizing pre-trained word embedding models leads to enhanced performance. One compelling analysis undercoring this advantage is the word analogy test (within the medical domain). It vividly illutrastes that pre-trained models talilored to specific domains consistently outperform their conterparts


### Context Retrival

In the development of question-answering systems, having a context for each question is often crucial. Typically, models are trained to understand and retrieve answers within a given context. However, our dataset contains only question and answers without any additional context.
This challenge presents a valuable opportunity for us to apply our knowledge to build a pseudo-document search engine. This search engine will enable us to retrieve relevant context for a given question, which can then be used for model training or other specific tasks.
The main steps of our process of context retrievel are:

*  **Keyword Extraction**: extract a set of kywords that capture the essence of the question using a pre-trained model, such as KeyBERT

* **Context Retrival**: using the extracted keywords as queries, we utilize the PubMedAPI to search for relevant documents through a vast repository of medical documents, extracting the most relevant ones.

* **Preprocessing and Cleaning**:  the retrieved contexts undergo preprocessing and cleaning to ensure they are in the appropriate format for futher analysis.

* **Context Ranking**: to prioritize the retrieved contexts, we employ various ranking techniques:

  *  **TF-IDF and BM25 Index Search**:  These methods assess the importance of terms based on their frequencies within the retrieved contexts and the question itself

  * **Document vectorization**: utilizing a TD-IDF vectorizer, we compute the textual documents into numerical vectors based on the relevancy of the words and how often they appear in the documents, enabling quantitative comparison

  * **Sentence Embedding and Cosine Similarity**: compute sentence embeddings for both the question and the retrieved contexts, then evaluate the cosine similarities between them to obtain similarity scores. This process allows us to rank the contexts based on their relevance to the question


The last method indeed yielded the best results, as it effectively maintains the semantic relationships between words. This capability enchances the matching process, enabling a more accurate alignment of the question with the context.
Using these methods, we were able to obtain sufficiently accurate contexts to associate with the questions. However, the effictiveness of this approach is limited by the quaility and reliablity of the external sources we use, as well as the performance of the models employed for document embeddings and the comparison between the documents and the query. Additionally, the computational complexity and time required for retriving and processing the context can also impact the efficiency of the system. Despite these limitations, this approach significantly enhances our ability to provide more relevant and contextually accurate answers



## 2. Training models:

### Approach for training the models:

After analysing the data we started research on various open source models that can be used for the project. At the beginning we faced a problem of not having the context for the question and answers in the dataset.

The dataset provided was different in terms of structure from the one that we saw in the tutorials. Hence had to search for models which can be fine tuned without the context.

After extensive reasearch we were able to find models like Gemma, Mistral, llama. We leveraged pre-trained versions of Gemma, Mistral, or llama for various QA tasks by fine-tuning them on our Medical datasets. This allowed us for efficient transfer learning, where the model learns from a large, general-purpose dataset and then adapts its knowledge to the specific task at hand.

Parallely, we also worked on retrieving the context for our dataset which has been explained in detail in the above section.

###Models

For our task we have deployed different solutions, and analyzed different architectures to complete the task. At first glance we can divide them into two groups:



*   Pretrained models fine-tuned on our dataset
*   Pretrained context-based models



#### Pretrained models fine-tuned on our dataset

The procedure followed in the training phase for all the models was the following:
* first the raw model was imported, specifying if needed that the task is Q-A
* we defined the hyperparameters for the training phase
* during fine-tuning on the dataset an evaluation set is always used to monitor the performances during training
* lastly performances w.r.t the raw model without fine-tuning are compared in order to assess if there were improvements thank to fine-tuning

**1**. **TinyLlama**

WE used TinyLlama and fine-tuned it on our medical dataset. Tiny-Llama is an LLM with only 1.1 billion parameters and is trained on 3 trillion tokens.
We used different techniques like LoRA and QLoRA to fine tune it.We used an open-source platform Unsloth for fine tuning our model because it is  387% faster + use 74% less memory on 1 epoch.


**2**. **Gemma (7B)**

We trained Gemma (7B) model using unsloth which is an optimzed framework tailored to refine the fine-tuning process for large language models (LLMs). Renowned for its swiftness and memory efficiency, Unsloth can fasten up to 30x training speed and with a notable 60% reduction in memory usage.

We used intelligent weight upcasting, a feature that curtails the necessity for upscaling weights during QLoRA, thereby optimizing memory usage.

Additionally, Unsloth leverages bfloat16 swiftly, improving the stability of 16-bit training and expediting QLoRA fine-tuning.


**3**.  **MISTRAL (7B)**

We again used Unsloth to fine-tune our Mistral model as it has optimized Triton kernels, manual autograds, etc, to speed up training. It is almost twice as fast as Huggingface and Flash Attention implementation.

We used Unsloth to fine-tune a Mistral-7b model on the Medical dataset over Colab’s Pro T4 which had higher ram utilized to train the model.



**4**. **Llama 3 (8B)**

We used unsloth for fine-tuning Llama3 which had 8 billion parameter. Due to unsloth we are able to fasten up the process to 30x training speed and with a notable 60% reduction in memory usage.


**5** **GPT-2**

The GPT2-medium version was used with 345 million parameters.It differs from other since it is based on the Transformer architecture, specifically the decoder-only version and is a general purpose model and can handle different language tasks across different domains.



#### Pretrained context-based models

**BioBert**

For this type of models a context is needed in the training phase and inference phase.
The **context** is a sequence of text from which the answer can be extracted, it was retrieved, as previously explained, from MedPub dataset using their API.

The difference in the performances can be explained since BioBert needs a *start_index* where the answer starts in the context and an *end_index* but we don't have these informations so it set to 0, secondly the results depend also on the quality of the context retrieved.



**Note:** all the models were compared and evaluated on the same score metric in order to assess their performances properly, and compare with their raw models without fine-tuning

### Testing and performance scores

Assessing the performances was a crucial point of our task, however due to the nature of the task and the data, some analysis was performed, in fact two questions both correct can be scored as very different, semantic of the phrase must be taken into consideration, but even important keywords  (medical terms) must be present in the answer if it's correct.


Source of our analysis: https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

<img src="Initial Analysis\images\65ae30bca9335d1c73650df0_metricsven (1).jpg" width="600px">

As shown in the diagram above completely statistical scores aren't a good metric to compare true and generated answers. For example **BLEU** score doesn't give us precise results and even similar answers but containg different terms are scored as very different: only the precision for each matching n-gram isn't a good metric.

For this reason different Model-based scores have been analyzed and **BertScore** resulted being the best choise since it takes into consideration both semantic and improve n-gram.

<img src= "Initial Analysis\images\bert_score.png" width="800px">

Key elements of BertScore are:

* **Contextual Embeddings**
* **Cosine Similarity**
* **Token Matching for Precision and Recall**
* **Importance Weighting**: Rare words’ importance is considered using Inverse Document Frequency (IDF)
* **Baseline Rescaling**: BERTScore values are linearly rescaled to improve human readability


Precision, Recall and F1 returned from BertScore were used for comparison and assesment.

Lastly even a lot of human evaluation was used both to evaluate the truthfulness and to verify if the sentece was written correctly


## 3. Testing **Results**:

 <img src= "Initial Analysis\images\scores.jpg" width="800px">

As per the above table we can see that the **MedTinyLlama** and **MedGemma** model are the Best models with **90.6** and **91.6** **F1 score** respectively, which was evaluated using the BertScore.

If we also check the cosine similarity of the best two models they are giving good results.

**MedTinyLlama:** 0.78

**MedGemma:** 0.81

We choose TinyLlama as the best model for us due to following reasons mentioned below:


*   The Execution Time of the TinyLlama is less compared to the MedGemma model which helps us to inference in less time.
*   TinyLlama, which boasts 1.1 billion parameters, its applications are mainly suited to environments where larger models might not be as feasible due to hardware limitations or greater efficiency.
* As the F1 Score of TinyLlamma is just 1% less than the MedGemma model we choose it to be our best model.


The best cosine similarity was from the model **MedMistral** as the result from the model where more similar in terms of embeddings.






## 4. Possible extensions:

### MedCortana - Voice Interactive System

As an additional extension, we developed a voice interactive system using text-to-speech (TTS) and speech-to-text (STT) models, improving accessibility and user experience. The main components are:

*  **Whisper Model for SST Technology**: Utilized for automatic speech recognition.

*  **Question-Answering System**: Based on a pre-trained large language model fine-tuned for medical question-answering tasks. It provides accurate and relevant answers to medical questions obtained through STT technology.

*  **Tacotron2 and WaveGlow Models for TTS Technologies**: Used to synthesize natural-sounding speech from raw transcripts, delivering the answers from the question-answering system without additional prosody information.

The main challenge, which also impacts the user experience, is the latency between question input and answer output. This latency includes:

*  **Automatic Speech Recognition**: Typically takes 2 to 10 seconds without the fast model and 1 second with the fast model.

*  **Generating the Answer with the Question-Answering Model:** Averages around 3.8 seconds.

*  **Synthesizing Natural-Sounding Speech from the Raw Answer**: Averages around 12 seconds.

These time references are based on the length of the questions in our dataset and will vary depending on the length of both the question and the answer.






