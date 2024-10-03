# Data augmentation

Check and use main data augmenation in NLP:
- Expanding Contractions;
- Sentence Shuffling;
- Dialects or Slangs;
- Noise Injection:
  - Synonym Replacement;
  - Random Insertion;
  - Random Deletion;
  - Random Swap;
- Paraphrasing - model: `facebook/bart-large-cnn`;
- Back Translation - model: `Helsinki-NLP/opus-mt-en-uk` and `Helsinki-NLP/opus-mt-uk-en`;
- Text Generation using Language Models - model GPT2;

FYI: with textaugment lib it is possible to apply most of the techniques presented in this notebook easily. But to understand better what it does let's reveal how it works.

**Expand Contraction** in NLP data augmentation refers to the process of expanding contractions (e.g., "don't" becomes "do not") to their full form. This can be useful to normalize text, as expanded forms are sometimes easier for models to interpret. It can also provide variety in training data for models.

**Sentence Shuffling** involves shuffling the order of sentences within a text or paragraph. The goal is to provide the model with slightly different versions of the original text to improve its robustness. However, shuffling should be applied carefully to maintain the overall coherence of the text, especially for tasks like language modeling or text classification.

Using different **Dialects or Slang** is a powerful NLP data augmentation technique where you alter the text by substituting formal or standard language with regional dialects, slang, or colloquialisms. This helps models become robust against linguistic variations that are common in real-world settings. It is particularly useful for chatbots, social media sentiment analysis, and any task where informal language is common.  
Challenges:
- Not all dialects or slang terms are easily replaceable, and some substitutions might alter the meaning of the sentence.
- You need a dictionary or set of rules for mapping standard language to slang or dialect variants.

**Noise injection** is a NLP data augmentation technique where noise is deliberately introduced into text data to improve model robustness. The idea is to simulate real-world imperfections or variations in text, such as typos, word swaps, or random deletions, allowing models to generalize better by training on slightly "noisy" data.  
Common Types of Noise Injection:
- Character-level noise: This involves adding noise at the character level, such as swapping adjacent characters, adding random characters, or deleting characters.
- Word-level noise: In this approach, noise is introduced at the word level by randomly deleting, swapping, or replacing words in the sentence.
- Synonym replacement: Words are replaced with their synonyms to maintain meaning while changing the wording slightly.
- Random deletion: Randomly deleting words from a sentence.
- Word swapping: Two adjacent words in a sentence are swapped.

**Paraphrasing** sentence rephrasing was done using BART model from HuggingFace.

**Back Translation** were done using models: `Helsinki-NLP/opus-mt-en-uk` and back to English using `Helsinki-NLP/opus-mt-uk-en`
DO NOT FORGET TO SUPPORT UKRAINE AGAINST F*** russians!!!

**Text generation** in the context of Natural Language Processing (NLP) data augmentation involves creating new text samples based on existing data. It is particularly useful when there is a limited amount of labeled data available for training machine learning models. By generating synthetic data, models can be trained on a more diverse set of inputs, which can improve their generalization ability.


# Classifiers

## Naive Bayes classifier

I choose MultinomialNB instead of ComplementNB here since it shows better results. Try it yourself, just comment on one model and uncomment another within Step 5 in the cell below.

Multinomial Naive Bayes (MultinomialNB)
It implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification (where the data are typically represented as word vector counts, although tf-idf vectors are also known to work well in practice).

Complement Naive Bayes (ComplementNB)
Designed for imbalanced data: ComplementNB was specifically created to address some of the weaknesses of MultinomialNB when dealing with imbalanced data. It focuses on the complement of each class, helping it perform better on minority classes.
How it works: Unlike MultinomialNB, which estimates the likelihood of each class, ComplementNB estimates the likelihood of not being in the class, which helps mitigate the dominance of the majority class.

More info at scikit-learn doc [link](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes).

## Support Vector Machine (SVM) classifier

We use the class_weight='balanced' since our dataset is imbalanced.  
We use a linear kernel for text classification, reasons:
- In text classification, the data often lies in high-dimensional sparse spaces (many features but most of them are zeros, e.g., from TF-IDF or CountVectorizer representations). A linear kernel can effectively separate classes in this space.
- The linear kernel is computationally much faster than non-linear kernels, especially when working with high-dimensional text data.
- In text classification problems like spam detection, the relationship between features (words) and the target (spam/ham) is often linear, which is why a linear kernel performs well.
