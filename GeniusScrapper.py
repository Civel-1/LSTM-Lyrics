import requests
import urllib.request
import urllib.parse
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

# How to scrap from new artist ? 
## GET /search : https://docs.genius.com/#search-h2  -> get artist id in response->hits->primary_artist->id
## then GET /artists/:id/songs : https://docs.genius.com/#artists-h2 -> get all path for songs.
## Then scrap with beautifoul soup

# -------------------- Some random artists
#19217 = Vald
#276476 = Sch
#1282 = Booba
#335710 = Pnl
#15427 = Alkpote

#69 = J-cole
#45 = Eminem

# ------------ Set artist to scrap lyrics (id, artist name)
artist = (69, 'J-cole')



base = f"https://api.genius.com/artists/{artist[0]}/songs?"
client_access_token = "bRbUdATtVwxD_2T08hElho95FrtMcnZz44h77x0w_hMcxO1nq57Rq7qR__urO5E4"

token = 'Bearer {}'.format(client_access_token)
headers = {'Authorization': token}

has_next_page = True
next_page = 1
prefix = f"/{artist[1]}"
all_urls = {}
while has_next_page:
    new_url = f"{base}per_page=50&page={next_page}"

    req = requests.get(new_url,headers=headers).json()  ## this should do GET request for the third page and so on...

    for i in range(0,len(req['response']['songs'])) :
        if req['response']['songs'][i]['path'].startswith(prefix) :
            all_urls[req['response']['songs'][i]['full_title']] = req['response']['songs'][i]['url']

    if req['response']["next_page"]!= None:
        next_page = req['response']["next_page"]
    else:
        has_next_page = False


fichier = open(f"lyrics/{artist[1]}_solo_lyrics.txt", "w")
for songs_title, urls in tqdm(all_urls.items(), "Writing lyrics") : 
    response = requests.get(urls, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36'})
    soup = BeautifulSoup(response.text, "lxml")
    lyrics = soup.findAll("div", {"class": "lyrics"})[0].text
    fichier.write(songs_title)
    fichier.write(lyrics)
fichier.close()