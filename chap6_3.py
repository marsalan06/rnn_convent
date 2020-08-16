from keras.preprocessing.text import Tokenizer
samples={"the cat sat on the mat","the dog ate my work","cat lol dog"}
tokens=Tokenizer(num_words=200)
tokens.fit_on_texts(samples) #internal list updated in memory , doesnt return any thing
sequence_list=tokens.texts_to_sequences(samples)
print(sequence_list) #[[2, 4, 3], [1, 2, 5, 6, 1, 7], [1, 3, 8, 9, 10]]
one_hot_result= tokens.texts_to_matrix(samples,mode='binary')
#print(one_hot_result) #list of vectors based on word index positions 
print(tokens.word_index) #{'the': 1, 'cat': 2, 'dog': 3, 'lol': 4, 'sat': 5, 'on': 6, 'mat': 7, 'ate': 8, 'my': 9, 'work': 10}