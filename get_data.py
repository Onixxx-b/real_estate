import requests
from bs4 import BeautifulSoup
import json

flats = []

for page in range(0, 596):
    url = 'https://rieltor.ua/kiev/flats-sale/?sort=-default&page=' + str(page)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for card in soup.select('.catalog-card'):
        flat = {}
        flat['price'] = card.select_one('strong.catalog-card-price-title').text.strip()
        flat['region'] = card.select_one('.catalog-card-region').text.strip()
        flat['room_count'] = card.select_one('.catalog-card-details-row:nth-of-type(1)').text.strip()
        flat['square'] = card.select_one('.catalog-card-details-row:nth-of-type(2)').text.strip()
        flat['floor'] = card.select_one('.catalog-card-details-row:nth-of-type(3)').text.strip()
        try:
            flat['subway'] = card.select_one('a.-subway').text.strip()
        except AttributeError:
            flat['subway'] = None
        flats.append(flat)

json_data = json.dumps(flats, ensure_ascii=False)

with open('data/flats_02_2024.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
