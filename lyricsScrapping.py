
##--## Get lyrics from top tracks of artists ##--##

import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# Set up client credentials
client_id = '7ce75114edf542fabcb8e8216d0067d6'
client_secret = '07c3769998194aed8450a36a2b71ae8d'
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

## Get top tracks of artist

def get_top_tracks(artist_id):
    results = sp.artist_top_tracks(artist_id)
    top_tracks = []
    for track in results['tracks']:
        top_tracks.append((track['name'],track['id'],track['external_ids']['isrc']))
    return top_tracks

#print(get_top_tracks('4tZwfgrHOc3mvqYlEYSvVi'))

##--## Get lyrics from top tracks, from musixmatch ##--##

key = "c1a664cb491a515186cb21424b020c1b"

def get_hook(track_isrc):
    url = f"http://api.musixmatch.com/ws/1.1/track.get?track_isrc={track_isrc}&apikey={key}"
    response = requests.request("GET", url)
    if response.status_code == 200: 
        return response.json()
    elif response.status_code == 401: 
        print("invalid/missing API key")
    elif response.status_code == 402: 
        print("The usage limit has been reached. Try to use another api key")
    elif response.status_code == 403: 
        print("You are not authorized to perform this operation")
    elif response.status_code == 404: 
        print("The requested resource was not found.")
    else: 
        print("Ops. Something were wrong.")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def get_soup(url: str) -> BeautifulSoup:
    '''Utility function which takes a url and returns a Soup object.'''
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup

def get_lyrics(url: str) -> str:
    '''Scrape a song url for it's lyrics using BeautifulSoup.'''
    soup = get_soup(url)
    cols = soup.findAll(class_="lyrics__content__ok", text=True)
    lyrics = None
    if cols:
        lyrics = "\n".join(x.text for x in cols)
    elif data := soup.find(class_="lyrics__content__warning", text=True):
        lyrics = data.get_text()
    return lyrics

def lyrics_from_id(id):
    '''Return a list of tuples (track_name, lyrics) for an artist id'''
    lyrics_list = []
    for track in get_top_tracks(id):
        isrc = track[2]
        url = get_hook(isrc)['message']['body']['track']['track_share_url']
        lyrics_list.append((track[0],get_lyrics(url)))
    return lyrics_list

#for song in lyrics_from_id('06HL4z0CvFAxyc27GXpf02'):
#   print(song[0])
#    print(song[1])
#    print('---')

#print(get_lyrics('https://www.musixmatch.com/fr/paroles/Eminem/Rap-God'))

## Transform into embedding

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#sentences = [get_lyrics('https://www.musixmatch.com/lyrics/Ed-Sheeran/Shape-of-You'), get_lyrics('https://www.musixmatch.com/fr/paroles/Karol-G/Ocean'),get_lyrics('https://www.musixmatch.com/lyrics/Lil-Nas-X/INDUSTRY-BABY'), get_lyrics('https://www.musixmatch.com/fr/paroles/Måneskin-1/I-WANNA-BE-YOUR-SLAVE'), get_lyrics('https://www.musixmatch.com/fr/paroles/PNL/Pnl')]
#sentences = [x for x in sentences if x is not None]


#model = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')
#embeddings = model.encode(sentences)
#print(embeddings)
#print(embeddings.shape)
#print(cosine_similarity(embeddings,embeddings))

## Perform tf-idf on the sentences on show cosine similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#vectorizer = TfidfVectorizer()
#vectors = vectorizer.fit_transform(sentences)

#print(vectors.shape)
#print(cosine_similarity(vectors, vectors))

#---------------------------------------------#

artists = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/allArtists60+.csv', sep=';').head(11)
artists = artists.drop(artists[artists['name'] == 'Eminem'].index)

def emb_from_id(id):
    lyrics = lyrics_from_id(id)
    sentences = [x[1] for x in lyrics if x[1] is not None]
    if len(sentences) == 0:
        return np.zeros(768)
    embeddings = model.encode(sentences)
    return np.mean(embeddings, axis=0)

def plot_emb():
    matrix = artists['embeddings'].to_numpy()
    matrix = np.vstack(matrix)
    print(matrix.shape)
    # Normalize
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # Remove Nan
    matrix = np.nan_to_num(matrix)
    from sklearn.metrics.pairwise import euclidean_distances
    euclidean = euclidean_distances(matrix,matrix)

    cosine_sim = cosine_similarity(matrix,matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(euclidean)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(artists['name'])))
    ax.set_yticks(np.arange(len(artists['name'])))
    # ... and label them with the respective list entries
    ax.set_xticklabels(artists['name'])
    ax.set_yticklabels(artists['name'])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(artists['name'])):
        for j in range(len(artists['name'])):
            text = ax.text(j, i, round(cosine_sim[i, j],2),
                            ha="center", va="center", color="w")

    ax.set_title("Euclidean similarity between artists")
    fig.tight_layout()
    plt.show()

def emb_from_id(id):
    pass


# Get the soup output of the page
page = 'https://www.musixmatch.com/artist/lil-wayne'

#soup = get_soup(page)

# Print string objects of the soup

#print(soup.prettify())

#artists['embeddings'] = artists['id'].apply(emb_from_id)
#print('Emb done')
#plot_emb()

import requests
 
# use to parse html text
from lxml.html import fromstring 
from itertools import cycle
import traceback
 
 
def to_get_proxies():
    # website to get free proxies
    url = 'https://free-proxy-list.net/' 
 
    response = requests.get(url)
 
    parser = fromstring(response.text)
    # using a set to avoid duplicate IP entries.
    proxies = set() 
 
    for i in range(1,20):
        ip = parser.xpath('//tbody/tr['+str(i)+']/td[1]/text()')
        port = parser.xpath('//tbody/tr['+str(i)+']/td[2]/text()')
        https = parser.xpath('//tbody/tr['+str(i)+']/td[7]/text()')
        #print(ip, port, https)
        
        # to check if the corresponding IP is of type HTTPS
        if https[0] == 'yes':
 
            # Grabbing IP and corresponding PORT
            proxy = ":".join([ip[0],port[0]])
 
            proxies.add(proxy)
    return proxies

proxies = to_get_proxies()
 
# to rotate through the list of IPs
proxyPool = cycle(proxies) 
 
# insert the url of the website you want to scrape.
url = 'https://www.musixmatch.com/artist/YOASOBI' 
 
for i in range(1, 11):
 
    # Get a proxy from the pool
    proxy = next(proxyPool)
    print("Request #%d" % i)
    print("Proxy: %s" % proxy)
 
    try:
        response = requests.get(url, proxies={"http": proxy, "https": proxy})
        print(response.json())
 
    except:
       
        # One has to try the entire process as most
        # free proxies will get connection errors
        # We will just skip retries.
        print("Skipping.  Connection error")