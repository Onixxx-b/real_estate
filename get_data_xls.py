import json
import xlwt

with open('flats_02_2024.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

workbook = xlwt.Workbook()
sheet = workbook.add_sheet('Flats')

columns = ['Price', 'Region', 'Room count', 'Square', 'Floor', 'Subway']

for i, column in enumerate(columns):
    sheet.write(0, i, column)

for i, flat in zip(range(len(data['flats'])), data['flats']):
    sheet.write(i+1, 0, flat['price'])
    sheet.write(i+1, 1, flat['region'])
    sheet.write(i+1, 2, flat['room_count'])
    sheet.write(i+1, 3, flat['square'])
    sheet.write(i+1, 4, flat['floor'])
    sheet.write(i+1, 5, flat['subway'])

workbook.save('flats_02_2024.xls')
