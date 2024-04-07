import pandas as pd

flats_data = pd.read_excel("data/flats.xls")

clean_flats_data = flats_data
clean_flats_data['Price'] = clean_flats_data['Price'].str.replace(" ", "").str.extract(r'(\d+)').apply(pd.to_numeric)
clean_flats_data['Room count'] = clean_flats_data['Room count'].str.extract(r'(\d+)').apply(pd.to_numeric)

clean_flats_data['Max Floor'] = clean_flats_data['Floor'].str.split(' ').str[3].apply(pd.to_numeric)
clean_flats_data['Floor'] = clean_flats_data['Floor'].str.split(' ').str[1].apply(pd.to_numeric)

clean_flats_data['Region'] = clean_flats_data['Region'].str.split(',').str[1].str.replace('р-н', '').str \
    .replace(" ", "").str.replace("\n", "")

clean_flats_data['Region'] = clean_flats_data['Region'].replace(
    {'Голосіївський': 1, "Солом'янський": 2, 'Шевченківський': 3, 'Святошинський': 4, 'Подільський': 5,
     'Дарницький': 6, 'Печерський': 7, 'Оболонський': 8, 'Деснянський': 9, 'Дніпровський': 10})

clean_flats_data['Living Square'] = clean_flats_data['Square'].str.split('/').str[1].str.replace(" ", "").apply(
    pd.to_numeric)

clean_flats_data['Kitchen Square'] = clean_flats_data['Square'].str.split('/').str[2].str.replace("м²", "").str.replace(
    " ", "").apply(pd.to_numeric)

clean_flats_data['Square'] = clean_flats_data['Square'].str.split('/').str[0].str.replace(" ", "").apply(pd.to_numeric)
clean_flats_data = clean_flats_data.rename(columns={'Square': 'Total Square'})

clean_flats_data['Subway'] = clean_flats_data['Subway'].apply(lambda x: 1 if pd.notnull(x) else 0)

clean_flats_data = clean_flats_data.dropna()
clean_flats_data = clean_flats_data.drop(clean_flats_data[clean_flats_data['Total Square'] > 300].index)

clean_flats_data.to_excel("clean_flats_02_2024.xlsx")
