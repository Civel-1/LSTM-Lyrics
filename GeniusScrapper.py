import requests
import urllib.request
import urllib.parse
import json
from bs4 import BeautifulSoup
from tqdm import tqdm


# -------------------- Some random Ids
#19217 = Vald
#276476 = SCH
#1282 = Booba
#335710 = PNL
#15427 = Alkpote

#69 = J.Cole
#45 = Eminem
#3927 = Phil Collins

# ------------ Set artist to scrap lyrics (id, artist name)
artist = (3927, 'Phil Collins')


all_urls = []
base = f"https://api.genius.com/artists/{artist[0]}/songs?"
client_access_token = "bRbUdATtVwxD_2T08hElho95FrtMcnZz44h77x0w_hMcxO1nq57Rq7qR__urO5E4"

token = 'Bearer {}'.format(client_access_token)
headers = {'Authorization': token}

has_next_page = True
next_page = 1
#suffix = f"\xa0{artsit[1]}"

while has_next_page:
    new_url = f"{base}per_page=50&page={next_page}"

    req = requests.get(new_url,headers=headers).json()  ## this should do GET request for the third page and so on...

    for i in range(0,len(req['response']['songs'])) :
        #if req['response']['songs'][i]['full_title'].endswith(suffix) :
        all_urls.append(req['response']['songs'][i]['url'])

    if req['response']["next_page"]!= None:
        next_page = req['response']["next_page"]
    else:
        has_next_page = False


fichier = open(f"{artist[1]}_lyrics.txt", "w")
for urls in tqdm(all_urls, "Writing lyrics") : 
    response = requests.get(urls, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36'})
    soup = BeautifulSoup(response.text, "lxml")
    lyrics = soup.findAll("div", {"class": "lyrics"})[0].text
    fichier.write(lyrics)
fichier.close()