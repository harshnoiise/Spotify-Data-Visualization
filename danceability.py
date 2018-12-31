import spotipy
import sys
import pprint
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from credentials import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cid = client_id
secret = client_secret
username = '127149407'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

scope = 'user-library-read playlist-read-private playlist-modify-public'
token = util.prompt_for_user_token(username, scope)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)


good_playlist = sp.user_playlist("127149407", "5IR1utIG0M7yoUegV7dSnc")
good_tracks = good_playlist["tracks"]
good_songs = good_tracks["items"]
while good_tracks['next']:
    good_tracks = sp.next(good_tracks)
    for item in good_tracks["items"]:
        good_songs.append(item)
good_ids = []
for i in range(len(good_songs) - 500):
    good_ids.append(good_songs[i]['track']['id'])

features = []
for i in range(0,len(good_ids),50):
    audio_features = sp.audio_features(good_ids[i:i+50])
    for track in audio_features:
        features.append(track)
        features[-1]['target'] = 1

trainingData = pd.DataFrame(features)

label = trainingData['danceability']
index = np.arange(len(label))

plt.bar(index, label)
plt.xlabel('songs', fontsize=10)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('danceability', fontsize=10)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Danceability of songs in my Spotify Library')
plt.show()
