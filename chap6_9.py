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
#print(x_train[133],y_train[133])

#6.10
#https://nlp.stanford.edu/projects/glove/
glove_dir="C:/Users/arsal/Desktop/anaconda/ml/quater 3/CHAP6/glove"
embeddings_index={}
file_glove=open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding="utf8") #open file
for line in file_glove: 
    values=line.split() #split after each line in list 
    word=values[0] #the first value of list is the word
    coefs=np.asarray(values[1:],dtype='float32') #next values in list is the vector 100 words
    embeddings_index[word]=coefs #saved the word and vector as key value pair, 400000 words
file_glove.close()


# print(line)
# print(values)
# print(word)

#6.11 preparing matrix for glove 
embedding_dim =100 #total vector length for each word
embedding_matrix = np.zeros((maxwords,embedding_dim)) #empty matrix of 10000,100
#10000 words for embedding space and 100 dimensions per word by glove
for word, i in word_index.items(): #item returns word and integer , enrich 24436, strayer 24437, vampiric 24438
    #print(word,i)
    if i< maxwords:
        embedding_vector= embeddings_index.get(word) #embeddings_index is dict words and vectors
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  #10000 words and 100 dimensions created in a matrix

#print(embedding_matrix[2:3,:]) #array sliced for 2nd row and all 100 columns

#6.12 model 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, Flatten, Dense
model= Sequential()
model.add(Embedding(maxwords,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation='sigmoid'))
#print(model.summary())

#stop traing of embedding layer , use glove trained matrix
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))
model.save_weights('pre_trained_glove_model.h5')

#importing matplotlib for visualization
import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

