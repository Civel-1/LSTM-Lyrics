import json
import urllib
from urllib.request import urlopen

base_url_get_song = "https://api.musixmatch.com/ws/1.1/matcher.lyrics.get?format=jsonp&callback=callback"
track_search_parameter = "&q_track="
artist_search_parameter = "&q_artist="
api_key = "&apikey=a0aa2a16a2ea9135a13e01e2f551dee5"


songs_name = ["HUMBLE.", "Bad Blood", "The Greatest", "Don\'t Wanna Know", "Freedom", "Pray For Me ",
              "LOVE. FEAT. ZACARI.", "DNA.",
              "Momma I Hit A Lick",
              "LOYALTY. FEAT. RIHANNA.",
              "Textbook Stuff",
              "Whose House",
              "The Heart 1-3",
              "Gangsta Party (Freestyle)",
              "The Heart Pt 3",
              "I\'m Ghost",
              "Watts R.i.o.t.",
              "Bitch, Don\'t Kill My Vibe",
              "F**kin\' Problems",
              "All The Stars",
              "King\'s Dead ",
              "m.A.A.d city",
              "ELEMENT.",
              "Poetic Justice",
              "Do It Again",
              "Hpnotiq"]

file = open("Kendrick.txt", "w")

for i in range(len(songs_name)):
    song = songs_name[i].replace(" ", "%20")
    api_call = base_url_get_song + track_search_parameter + song + artist_search_parameter + "Kendrick%20Lamar" + api_key

    print(api_call)
    uh = urllib.request.urlopen(api_call)
    data = uh.read()
    data_str = data.decode("utf-8")
    data_final = data_str[9:-2]
    JSON_object = json.loads(data_final)
    print(JSON_object)
    lyrics = JSON_object['message']['body']['lyrics']['lyrics_body']
    lyrics_final = lyrics[:-58]
    lyrics_final.replace('\'', "'")
    print(lyrics_final)
    file.write(lyrics_final)

