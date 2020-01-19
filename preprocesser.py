from emoji import UNICODE_EMOJI
import pandas as pd
import string

def contains_emoji(text):
    for e in UNICODE_EMOJI.keys():
        if e in text:
            return True
    return False

# load datasets
print('Loading Datasets...')
twcs = pd.read_csv('data/twitter/twcs.csv')
twair = pd.read_csv('data/twitter/airlines.csv')
print('Finished')
# strip text out of each dataset - use 'text' column
print('Stripping Text Data...')
twcs = twcs[['text']]
twair = twair[['text']]
print('Finished')
# merge datasets
print('Merging Datasets...')
text_data = twcs
text_data = text_data.append(twair)
print('Finished')
# filter to only include text with emojis
print('Getting Relevant Data...')
text_data = text_data[text_data.text.apply(contains_emoji)]
text_data = text_data.reset_index(drop=True)
print('Finished')
# strip out hashtags
print('Cleaning Text')

def strip_hashtags(text):
    return ' '.join([w for w in text.split(' ') if '#' not in w])

def strip_punctuation(text):
    return ''.join([ch for ch in text if ch not in string.punctuation])

text_data['text'] = text_data.text.apply(strip_hashtags)
text_data['text'] = text_data.text.apply(strip_punctuation)
print('Finished')


text_data.to_csv('data/preprocessed/text_data.csv')
