
## 2nd try

import lyricsgenius as genius
import pandas as pd
import string 
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')   


access_token = 'jRNoIpNbQ3039JwG9vgdh6VMpgklwWkrB7ATgx9sswat5vK5ZgFJPGcM5Ol3PCw8'

def add_data(df,m,access_token,done=0):
    """
    This function uses the library lyricsgenius to extract the fields
    title, artist, album, date and lyrics and stores them into a pandas dataframe

    parameters:
    query = artist or band to search
    n = max numbers of songs
    access_token = your access token of the genius api
    """
    
    api = genius.Genius(access_token)

    df['titles'] = ''
    df['lyrics'] = ''
    n = len(df['name'])
    lyrics = []
    titles = []

    for i,artist in enumerate(df['name']):
        if i>done :
            list_lyrics = []
            list_title = []

            artist = api.search_artist(artist,max_songs=m,sort='popularity')
            songs = artist.songs
            for song in songs:
                list_lyrics.append(song.lyrics)
                list_title.append(song.title)

            lyrics.append(list_lyrics)
            titles.append(list_title)

            print(i, '/', n)
            if i%10 == 0 or True:
                # Update the dataframe, beginning from the line done+1
                df['titles'][done+1:i+1] = titles
                df['lyrics'][done+1:i+1] = lyrics
                df.to_csv('/Users/romaindufly/Desktop/Projet MODAL/geniusNew.csv', sep=';')
                print("Saved!")
    print("Ended process!")

def clean_array(arr):
    new_arr = []
    for e in arr :
        e = e.split("\n")[1:-1]
        e = " ".join(e)
        e = re.sub(r"\[([^]]+)\]", '', e)
        e = e.strip().lower()
        new_arr.append(e)
    return new_arr

def clean_lyrics(df,column):
    # For every row in the column, for every element in the array, clean the lyrics
    for i in range(len(df[column])):
        if df[column][i] != 0:
            df[column][i] = clean_array(df[column][i])
    return df

def lyrics_to_words(document):
    """
    This function splits the text of lyrics to single words, removing stopwords and doing the lemmatization to each word

    parameters:
    document: text to split to single words
    """
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stop_words])
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())
    return normalized

df = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/genius.csv', sep=';')
df = add_data(df,10,access_token, 201)

#df = clean_lyrics(df,'lyrics')

#df.to_csv('/Users/romaindufly/Desktop/Projet MODAL/genius.csv', sep=';')