import numpy as np 
samples=["my name is arsalan", "arsalan is my name","arsalan come here my boy"]
#"starting a new coding practice","this is word level one hot encoding"]
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word]=len(token_index)+1

#print(token_index) #token index dictionary
max_len=10
token_vector=np.zeros(shape=(len(samples),max_len,max(token_index.values())+1)) #z,y,x
print(token_vector)
for i, sample in enumerate(samples):
    #print(i,sample) # 0 my name is arsalan 1 i am an engineer
    #print(list(enumerate(sample.split()))[:max_len])
    for j, word in list(enumerate(sample.split()))[:max_len]:
        index = token_index.get(word)
        #print(index,word) #index 4, word arsalan
        token_vector[i, j, index] = 1.
print(token_vector)