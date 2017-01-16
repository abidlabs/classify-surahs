
# Learning to Classify Quranic Chapters as 'Meccan' or 'Medinan'

In this notebook, we explore whether it is possible to build a binary classifier for surahs based on the words used in their verses. 

## Load the Dataset


```python
import numpy as np, pandas as pd

verses = pd.read_csv("data/verses.txt", header=0, delimiter="|", quoting=3, encoding='utf-8')
labels = np.genfromtxt("data/surah-labels.csv", delimiter=",")
```

## Convert to Bag-Of-Words Representation

This step can take up to a minute or so to run. It can be optimized but its not necessary for a fixed corpus, like the text of the Quran


```python
# -*- coding: utf-8 -*-
n_verses = verses.shape[0]
word_bag = list()
valid_verse_counter = 0

for verse in range(n_verses):
    if type(verses["Text"][verse]) is str:
        valid_verse_counter += 1
        for word in verses["Text"][verse].split(" "):
            word_bag.append(word)
            
print("The Quran has",len(word_bag)," words.")
word_bag = list(set(word_bag))
print("The Quran has",len(word_bag),"unique words, allowing for vowel variations.")

word_vectors = np.zeros(shape=[valid_verse_counter,len(word_bag)])
verse_labels = np.zeros(shape=[valid_verse_counter])-1
verse_chapter_mapping = np.zeros(shape=[valid_verse_counter]) 

valid_verse_counter = 0

for verse in range(n_verses):
    if type(verses["Text"][verse]) is str:
        for word in verses["Text"][verse].split(" "):
            index = word_bag.index(word)
            word_vectors[valid_verse_counter,index] += 1
        #apply labels to individual verses
        ch_number = int(verses["Chapter"][verse])
        verse_labels[valid_verse_counter] = labels[ch_number-1,1] #-1 for 0-indexed np array
        #create a mapping between verses and chapters
        verse_chapter_mapping[valid_verse_counter] = ch_number

        valid_verse_counter += 1
        
print("The Quran has approximately",np.count_nonzero(verse_labels)," Meccan verses.")
word_bag = list(set(word_bag))
print("The Quran has approximately",len(verse_labels)-np.count_nonzero(verse_labels)," Medinan verses.")
print("This is approximate because some Meccan surahs include Medinan verses and vice versa")
```

    The Quran has 78245  words.
    The Quran has 14870 unique words, allowing for vowel variations.
    The Quran has approximately 4613  Meccan verses.
    The Quran has approximately 1623  Medinan verses.
    This is approximate because some Meccan surahs include Medinan verses and vice versa
    

## Partition Training and Validation


```python
#split based on surah level (to preserve independence between training and validation set)
def partition(features, labels, verse_chapter_mapping, train_fraction=0.4):
    n_chapters = int(np.max(verse_chapter_mapping))
    n_train = int(train_fraction*n_chapters)
    n_valid = n_chapters - n_train
    
    chapters = np.random.permutation(n_chapters) + 1 #zero-indexed np array
    train_chapters, valid_chapters = chapters[:n_train], chapters[n_train:]
    
    train_indices = np.where(np.in1d(verse_chapter_mapping,train_chapters))[0]
    valid_indices = np.where(np.in1d(verse_chapter_mapping,valid_chapters))[0]
    
    features_train = features[train_indices]
    labels_train = labels[train_indices]
    features_valid = features[valid_indices]
    labels_valid = labels[valid_indices]
    
    return (features_train, labels_train, features_valid, labels_valid)

features_train, labels_train, features_valid, labels_valid = partition(word_vectors, verse_labels, verse_chapter_mapping)
print("The training set has:",len(labels_train),"verses")
print("The validation set has:",len(labels_valid),"verses")
```

    The training set has: 2357 verses
    The validation set has: 3879 verses
    

## Logistic Regression


```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model = model.fit(features_train, labels_train)
```

## Results

#### Verse Level


```python
# check the accuracy on the training set
print('We achieve', str(round(100*model.score(features_train, labels_train), 2))+'% accuracy on the training set')
print('We achieve', str(round(100*model.score(features_valid, labels_valid), 2))+'% accuracy on the validation set')
```

    We achieve 98.68% accuracy on the training set
    We achieve 86.26% accuracy on the validation set
    

Given the large number of features, its not surprising that there is some overfitting happening. However, our results on the validation set do suggest that the classifier is learning something. How much better is this than chance? Well, we started with an unbalanced dataset, so if our classifier was just classifying every verse as Meccan, it would get an accuracy of about 74%. Depending on the specific partitioning of the training and validation, this is around 10 percentage points higher than that. 

#### Surah Level

We can also see how well the classifier can predict an entire *surah* is Meccan or Medinan. This is also more fair since many Meccan surahs include Medinan verses and vice versa, which we don't take into accout when training or measuring the performance of our results.

We will simply use majority voting to decide if a surah is Meccan or Medinan. If a majority of a surahs' verses are Meccan, the entire surah will be Meccan and same for Medinan. 


```python
chapter_predictions = np.zeros(shape=[int(np.max(verse_chapter_mapping)), 2])-1
for chapter in range(1,int(np.max(verse_chapter_mapping))+1):
    #get all the verse of that chapter
    verses = np.where(verse_chapter_mapping==chapter)[0]
    verse_predictions = model.predict(word_vectors[verses])
    #see what majority of voters say
    chapter_pred = np.round(np.mean(verse_predictions))
    chapter_conf = np.mean(verse_predictions)
    chapter_predictions[chapter-1, 0] = chapter_pred
    chapter_predictions[chapter-1, 1] = round(100*(0.5+abs(0.5-chapter_conf)) 
    
#compare to actual chapter labels
pred_errors = chapter_predictions[:,0] - labels[:,1]
print("Surahs It Misclassified as Meccan:")
print("---------------")
for i in np.where(pred_errors>0)[0]:
    print(i+1, " -- % Verses:", str(chapter_predictions[i,1]))
print("\nSurahs It Misclassified as Medinan:")
print("---------------")
for i in np.where(pred_errors<0)[0]:
    print(i+1)
```

    Surahs It Misclassified as Meccan:
    ---------------
    13  -- % Verses: 81.0
    22  -- % Verses: 63.0
    47  -- % Verses: 53.0
    55  -- % Verses: 100.0
    76  -- % Verses: 97.0
    99  -- % Verses: 100.0
    
    Surahs It Misclassified as Medinan:
    ---------------
    

#### Examining the Model

Further insight can be obtained by examining the weights in the logistic regression model. For example, we can see what words are most associated with a a Meccan surah, and what words are associated with a Medinan surah.


```python
TOP_N = 10
weights = model.coef_.flatten()

idx = np.argpartition(weights, -TOP_N)[-TOP_N:]
print("Top Meccan Words\n------------------")
for i in idx:
    print(word_bag[i])

idx = np.argpartition(weights, TOP_N)[:TOP_N]
print("Top Medinan Words\n------------------")
for i in idx:
    print(word_bag[i])
```

    Top Meccan Words
    ------------------
    بعهدكم
    وتقسطوا
    العالمون
    أولاهما
    لتحصنكم
    العمى
    نور
    لمستقر
    لنفد
    تنزيل
    Top Medinan Words
    ------------------
    تقتلني
    والله
    أكلها
    وإذ
    وبكفرهم
    مبصرة
    زلزلة
    بآية
    سخرناها
    وجهرا
    


```python
def weight_of_word(word):
    i = word_bag.index(word)
    w = model.coef_.flatten()[i]
    mx = max(abs(max(model.coef_.flatten())), abs(min(model.coef_.flatten())))
    prob = (w + mx) / (2*mx)
    print("The weight of this word is:", w)
    print("A very rough probability that the word is Meccan:", prob)
    
weight_of_word("الشمس")
```

    The weight of this word is: 0.164471590085
    A very rough probability that the word is Meccan: 0.536413190447
    
