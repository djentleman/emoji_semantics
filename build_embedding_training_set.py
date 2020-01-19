import pandas as pd
from emoji import UNICODE_EMOJI
import random
import json
import re

text_data = pd.read_csv('data/preprocessed/text_data.csv', engine='python')

# get list of unique words

contexts = {}

context_size = 4

corpus = text_data.text.values

lim = -1

def split_out_emojis(words):
    reconstructed = []
    for word in words:
        word_is_emoji = False
        reconstructed_word = ''
        for ch in word:
            if ch in UNICODE_EMOJI.keys():
                reconstructed.append(ch)
                word_is_emoji = True
            else:
                reconstructed_word += ch
        if not word_is_emoji:
            reconstructed.append(word)
        else:
            if reconstructed_word != '':
                reconstructed.append(reconstructed_word)
    return reconstructed

digits = re.compile('^\d*$')

for k, row in enumerate(corpus):
    if k == lim:
        break
    if row == None:
        continue
    words = str(row).split(' ')
    words = [w.lower() for w in words if w != '']
    words = [w for w in words if digits.match(w) is None]
    words = split_out_emojis(words)
    # split out any continous emoji strings
    for i, word in enumerate(words):
        for j in range(-context_size, context_size+1):
            context_idx = i + j
            if context_idx != i and context_idx >=0 and context_idx <= (len(words)-1):
                context = words[context_idx]
                if word in contexts.keys():
                    contexts[word] |= {context}
                else:
                    contexts[word] = {context}


all_words = list(contexts.keys())

mapping = {h: i for i, h in enumerate(all_words)}
reverse_mapping = {str(i): h for i, h in enumerate(all_words)}
open('data/mapping.json', 'w+').write(json.dumps(mapping))
open('data/reverse_mapping.json', 'w+').write(json.dumps(reverse_mapping))

dataset = []

# build context dataset
for word, context_words in contexts.items():
    # add all the context words into the dataset
    w_id = mapping[word]
    for context_word in context_words:
        c_id = mapping[context_word]
        dataset.append([w_id, c_id, 1])
    # now add non context words
    for i in range(len(context_words)):
        while True:
            r_word = random.choice(all_words)
            if r_word in context_words:
                continue
            if r_word == word:
                continue
            break
        r_id = mapping[r_word]
        dataset.append([w_id, r_id, 0])

df = pd.DataFrame(dataset, columns=['word_a', 'word_b', 'has_context'])

df.to_csv('data/preprocessed/embedding_dataset.csv', index=None)

