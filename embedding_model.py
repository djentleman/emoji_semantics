import numpy as np
import pandas as pd
from keras.layers import (
    Input,
    Embedding,
    Reshape,
    Dense,
    dot,
)
from keras.callbacks import LambdaCallback
from keras.models import Model
from functools import reduce
from random import (
    choice as rndchoice,
    randint
)
import json
# train model

mapping = json.loads(open('data/mapping.json').read())
reverse_mapping = json.loads(open('data/reverse_mapping.json').read())

df = pd.read_csv('data/preprocessed/embedding_dataset.csv')

pairs = df[['word_a', 'word_b']].values
labels = df[['has_context']].values

vocab_size = len(list(mapping.keys()))

# network hyperparameters
vector_dim = 300
epochs = 500
batch_size = 500000




input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# get cosine similarity
similarity = dot(axes=0, inputs=[target, context], normalize=True)
# get dot product
dot_product = dot(axes=1, inputs=[target, context], normalize=False)
dot_product = Reshape((1,))(dot_product)

output = Dense(1, activation='sigmoid')(dot_product)

model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

similarity_model = Model(input=[input_target, input_context], output=similarity)

vector_model = Model(input=input_target, output=target)

def save_model(model, path):
    # serialize model to JSON
    print("Saving Model...")
    model_json = model.to_json()
    with open("%s.json" % path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("%s.h5" % path)
    print("Saved model to disk")

# this will probably need to me moved somwhere else...
def load_model(path):
    # load json and create model
    json_file = open('%s.json' % path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s.h5" % path)
    print("Loaded model from disk")
    return loaded_model

# build similarity lambda, and fitting function :)
def on_epoch_end(epoch, logs):
    print('Epoch: %d' % (epoch+1))
    if epoch % 20 != 19:
        print('skipping vaidation...')
        return
    words = [randint(0, vocab_size-1) for i in range(5)]
    save_model(vector_model, 'data/model/word_embedder')
    return
    for word in words:
        similarities = [(i, similarity_model.predict([np.array([word]), np.array([i])])[0][0][0]) for i in range(vocab_size)]
        similarities = sorted(similarities, key=lambda x: x[1])
        most_similar = similarities[-20:-1]
        least_similar = similarities[:19]
        print('------------------------------')
        print('Word: %s' % reverse_mapping[str(word)])
        print('Similar Words: %s' % ', '.join([reverse_mapping[str(x[0])] for x in most_similar]))
        print('Disimilar Words: %s' % ', '.join([reverse_mapping[str(x[0])] for x in least_similar]))
        print('------------------------------')




print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
on_epoch_end(-1, None)
model.fit([pairs[:, 0], pairs[:, 1]], labels, batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[print_callback])

save_model(vector_model, 'data/model/word_embedder')
