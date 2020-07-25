from keras.preprocessing.text import Tokenizer
samples={"the cat sat on the mat","the dog ate my work","cat lol dog"}
tokens=Tokenizer(num_words=200)
tokens.fit_on_texts(samples) #internal list updated in memory , doesnt return any thing
sequence_list=tokens.texts_to_sequences(samples)
#print(sequence_list)
one_hot_result= tokens.texts_to_matrix(samples,mode='binary')
print(one_hot_result) #list of vectors based on word index positions 
print(tokens.word_index) #dict of word to letters