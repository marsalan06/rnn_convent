import tensorflow as tf
from tensorflow.keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing 

#embedding_layer=Embedding(1000,64)

max_features =1000 #no of words to consider
maxlen=70 #max no of words in max_features 
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
#print(x_train[10])
#print("complete sentence in integer mode")

# word_to_id = imdb.get_word_index()
# id_to_word = {value:key for key,value in word_to_id.items()}     # get words of the integers
# print(' '.join(id_to_word[id] for id in x_train[10] ))
# print("english sentence")

x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen) #returns the whole list truncated to maxlen=20
x_test= preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)
#print("padded seq")
#print(x_train[10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense

model=Sequential()
model.add(Embedding(10000,10,input_length=maxlen))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
#print(model.summary())
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2)
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Accuracy: %f' % (accuracy*100))

#to make prediction
from keras.preprocessing.text import one_hot
d="what a great movie that was mate!"
encoded_docs = [one_hot(d, 1000)]
print(encoded_docs)
padded_docs=preprocessing.sequence.pad_sequences(encoded_docs,maxlen=70)
print(padded_docs)
print(model.predict(padded_docs))
print(model.predict_classes(padded_docs))


