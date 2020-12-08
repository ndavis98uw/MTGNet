import requests as req
from PIL import Image
import io
import time
import numpy as np
image_directory = 'card_images'
cards = req.get('https://api.scryfall.com/cards/search?order=cmc&q=t%3Acreature').json()
ids = []
names = []
colors = []
cmc = []
types = []
raw_directory = 'raw_data'
seen_card = True
while cards['has_more']:
    for card in cards['data']:
        if 'colors' in card.keys() and '—' in card['type_line']:
            print(card['name'])
            ids.append(card['id'])
            names.append(card['name'])
            colors.append(card['colors'])
            cmc.append(card['cmc'])
            types.append(card['type_line'][list(card['type_line']).index('—')+1:].split())
            # bytes = req.get(card['image_uris']['art_crop'])
        #     if not seen_card:
        #         image = Image.open(io.BytesIO(bytes.content))
        #         image.save(image_directory + '/' + card['id'] + '.png')
        #         time.sleep(.07)
        #         print('Imaged Saved')
        # if card['name'] == "Mockery of Nature":
        #     seen_card = False
    cards = req.get(cards['next_page']).json()
    time.sleep(.07)
for card in cards['data']:
    if 'colors' in card.keys() and '—' in card['type_line']:
        print(card['name'])
        ids.append(card['id'])
        names.append(card['name'])
        colors.append(card['colors'])
        cmc.append(card['cmc'])
        types.append(card['type_line'][list(card['type_line']).index('—')+1:].split())
        # bytes = req.get(card['image_uris']['art_crop'])
        # if not seen_card:
        #     image = Image.open(io.BytesIO(bytes.content))
        #     image.save(image_directory + '/' + card['id'] + '.png')
        #     time.sleep(.07)
        #     print('Imaged Saved')
    # if card['name'] == "Mockery of Nature":
    #     seen_card = False
np.save(raw_directory + '/ids.npy', ids)
np.save(raw_directory + '/names.npy', names)
np.save(raw_directory + '/colors.npy', colors)
np.save(raw_directory + '/cmc.npy', cmc)
np.save(raw_directory + '/types.npy', types)


