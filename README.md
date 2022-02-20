# Deep_learning_M2MO

# Introduction and objectif
This repositry is for my DMs and project for the deep learning lesson
 The goal of this project is to implement and test various approaches to text generation: starting from simple Markov Chains, through neural networks (LSTM): word-based approach and char-based approach. 
 The particularity of this generation task is that the lyrics are rhythmic yet at the same time are not necessarily meaningful. 
The goal is to produce a readable rhythmic text that is so much similar to the original singer lyrics.

There are so many approaches in generating text from which I decided to work with the above techniques. In fact,  these existing approaches have some deficiencies. For example, the word-based method did produce meaningful words yet not necessarily rhythmic and the LSTM-char- based model is totally the inverse. Markov chain is a simple but basic method that produces a text that is not necessarily fully comprehensive.  The challenge is to find the most suitable architecture that satisfies our goal.


# Exploratory data analysis
For the EDA part, it’s well known in text treatment to remove stop words, punctuation and do some treatment like lemming or stemming. Yet, in this project we try to generate comprehensive text, and that wouldn’t be achieved without stop words.

Training on cleaned word sequences produced poetry that didn’t quite make sense;  The following article discusses this idea with more details.
https://towardsdatascience.com/generative-poetry-with-lstm-2ef7b63d35af

# Modelling

In this part, I was interested in implementing different methods from different sources:
The first method was the Markov chain which I explained each step in the notebook.
The second model is the char-based model, where I followed a tutorial but I improved the architecture and factorized the code.
The third model is the one I propose, where I use char-based LSTM model and exploit the fact that this model can capture the rhythm in the lyrics so I used the fuzzy match algorithm to correct the misspelled generated words and find the wanted word using Markov chain method. I fusionned the Markov chain model and the LSTM-char-based model to have more accurate and readable generated text.

1- Use char-base model in order to produce rhythmic text 
2- Pass through the generated words, if the word doesn’t exist in the corpus of the lyrics, means that this word doesn’t exist:
     2.1 Interpret its phonetic using fuzzy algorithm
     2.2 Look using Markov model the most suitable word that can replace it:
              go through the proposed words by Markov model ( this model proposes a word by analyzing the word that just before) and find the most rhythmic similar word in this set using levenshtein distance. But, in case the distance were so big, we are going to look for the synonyms of the word proposed by Markov chain. Then we select from this set the most similar phonetically word and choose it.
The fourth model was the word based LSTM model, which is a standard LSTM with embedding layer, where I tried different architectures of LSTM and analyzed the result in order to select the most suitable architecture for our problem. I tweak the hyperparameters of my LSTM model, e.g. by using additional layers or nodes, or by trying out different optimizers. 

 


Char based model https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/ 

 Generating Rhyming Poetry Using LSTM Recurrent Neural Networks : https://dspace.library.uvic.ca/bitstream/handle/1828/10801/Peterson_Cole_MSc_2019.pdf?sequence=3&isAllowed=y

Markov chain Model
https://www.kaggle.com/paultimothymooney/poetry-generator-rnn-markov

Phonetic Algorithms Explained
https://stackabuse.com/phonetic-similarity-of-words-a-vectorized-approach-in-python/

Fuzzy matching
 https://www.informit.com/articles/article.aspx?p=1848528



# Evaluation metrics 

 Evaluating the models was a real  challenge in this project as there doesn't exist a defined metric for this problem ‘ Poetry generation’ although the range of metrics proposed for ‘Text generation’ and its different subfields that are described in the following survey. 
So, by identifying the problem of the outputs of my models I select a precise range of metrics.
The problem with my models’ outputs was the READABILITY of generated text, going from the Markov chain to the deepest LSTM proposed model. 
Even thought, the model is able to generate word, it doesn’t necessary makes sense, without talking about char-based model, where generated text contain inexisting words.
I choose to evaluate generated text using this readability indices: 

The automated readability index (ARI) is a readability test for English texts, designed to gauge the understandability of a text [3]. 

Flesch Reading Ease Score gives a text a score between 1 and 100, with 100 being the highest readability score. Scoring between 70 to 80 is equivalent to school grade level 8. This means text should be fairly easy for the average adult to read [1].

The Flesch–Kincaid readability tests are readability tests designed to indicate how difficult a passage in English is to understand [4] . 
![image](https://user-images.githubusercontent.com/99397991/154858706-7cafcef4-f4a4-4d6a-a8a0-887c227163ed.png)


Using these metrics, the goal is to make a text as readable as the original text, because even the original text - lyrics- are not so readable. 
readability: https://pypi.org/project/readability/
A Survey of Evaluation Metrics Used for NLG Systems
     https://arxiv.org/pdf/2008.12009.pdf




# Next steps
In addition to the observations above, there are several other possible directions for this project in the future:
Currently, the model uses a starting word as the seed for generating the rest of the poem. I could use additional seeds to improve relevance e.g. sentiment, themes or pictures.
I could also use additional features of the poems to improve the quality of the model’s output, e.g. syllables or part-of-speech.

Finally I could add more poetry to my dataset, and/or remove low-quality poetry from the dataset. This last point is very subjective, as different people can well have different perspectives of what makes a good poem.
I create a rhythm metric based on the leverstein distance and pronunciation library to measure the average distance of the first words and the last words of each line of the poetry or calculate the Entropy of the text.

Introducing Aspects of Creativity in Automatic Poetry Generation : https://aclanthology.org/2019.icon-1.4.pdf





# ** Sources** 


1* Flesch Reading Ease https://readable.com/readability/flesch-reading-ease-flesch-kincaid-grade-level/#:~:text=What%20is%20a%20Flesch%20Reading,being%20the%20highest%20readability%20score.&text=This%20means%20text%20should%20be,the%201940s%20by%20Rudolf%20Flesch.


3* ARI: Automated readability index designed to gauge the understandability of a text : https://en.wikipedia.org/wiki/Automated_readability_index


4*  Kincaid_readability_tests: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
