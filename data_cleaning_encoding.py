import pandas as pd
import numpy as np

flats_data = pd.read_excel("data/flats.xls")

clean_flats_data = flats_data
# print(clean_flats_data['Price'].isnull().sum())
# print(clean_flats_data['Room count'].isnull().sum())
# print(clean_flats_data['Floor'].isnull().sum())
# print(clean_flats_data['Square'].isnull().sum())
# print(clean_flats_data['Subway'].isnull().sum())


clean_flats_data['Price'] = clean_flats_data['Price'].str.replace(" ", "").str.extract(r'(\d+)').apply(pd.to_numeric)
clean_flats_data['Price'] = clean_flats_data['Price'] - clean_flats_data['Price'] * 0.1
clean_flats_data['Room count'] = clean_flats_data['Room count'].str.extract(r'(\d+)').apply(pd.to_numeric)

clean_flats_data['Max Floor'] = clean_flats_data['Floor'].str.split(' ').str[3].apply(pd.to_numeric)
clean_flats_data['Floor'] = clean_flats_data['Floor'].str.split(' ').str[1].apply(pd.to_numeric)

clean_flats_data['Living Square'] = clean_flats_data['Square'].str.split('/').str[1].str.replace(" ", "").apply(
    pd.to_numeric)

clean_flats_data['Kitchen Square'] = clean_flats_data['Square'].str.split('/').str[2].str.replace("м²", "").str.replace(
    " ", "").apply(pd.to_numeric)

clean_flats_data['Square'] = clean_flats_data['Square'].str.split('/').str[0].str.replace(" ", "").apply(pd.to_numeric)
clean_flats_data = clean_flats_data.rename(columns={'Square': 'Total Square'})

clean_flats_data['Subway'] = clean_flats_data['Subway'].apply(lambda x: 1 if pd.notnull(x) else 0)

clean_flats_data = clean_flats_data.dropna()
clean_flats_data = clean_flats_data.drop(clean_flats_data[clean_flats_data['Total Square'] > 300].index)

clean_flats_data['Region'] = clean_flats_data['Region'].str.split(',').str[1].str.replace('р-н', '').str \
    .replace(" ", "").str.replace("\n", "")
clean_flats_data = clean_flats_data.dropna(subset=['Region'])
# print(clean_flats_data['Region'].isnull().sum())
# print(clean_flats_data['Region'].unique())
clean_flats_data = pd.get_dummies(clean_flats_data, columns=['Region'], dtype=np.float32)

clean_flats_data.to_excel("data/clean_flats_encoding_minus_percent.xlsx")

print(clean_flats_data.columns.tolist())\
