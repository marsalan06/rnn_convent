import os
ibdm_dir="C:/Users/arsal/Desktop/anaconda/ml/quater 3/CHAP6/aclImdb"
train_dir=os.path.join(ibdm_dir,'train') #train directory
labels=[]
texts=[]

for label_type in ['neg','pos']: #two types
    dir_name=os.path.join(train_dir,label_type) #train directory/neg or pos
    for fname in os.listdir(dir_name): #find all files in folder
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname),encoding="utf8") #read file
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
#print(texts[20000],labels[20000])
#print(len(labels))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen=100 #max words from a review
train_samp=200
val_samp=10000
maxwords=10000 #max 10000 words from embedding space dictionary

tokenizer=Tokenizer(num_words=maxwords)
tokenizer.fit_on_texts(texts)
sequence= tokenizer.texts_to_sequences(texts) #sequence of words index in texts
word_index= tokenizer.word_index #word index as per embedding space on texts
#print(sequence)

data=pad_sequences(sequence, maxlen=maxlen) #truncating texts to 100 words per text
labels=np.asarray(labels)
#print(labels.shape,data.shape) #labels (25000,),data(25000,100) ... 25000 reviews 100 words each
 
indices=np.arange(data.shape[0]) #make a list 0-24900
np.random.shuffle(indices) #shuffle the list
data=data[indices] #save old data variable according to shuffled list
labels=labels[indices]#save old labels variable according to shuffled list

#spliting the data
x_train=data[:train_samp]
y_train=labels[:train_samp]
x_val=data[train_samp:train_samp+val_samp]
y_val=labels[train_samp:train_samp+val_samp]
print(x_train[133],y_train[133])



