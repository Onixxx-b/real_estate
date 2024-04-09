import pandas as pd

flats_data = pd.read_excel("data/flats_04_2024.xls")

clean_flats_data = flats_data
clean_flats_data['price'] = clean_flats_data['price'].str.replace(" ", "").str.extract(r'(\d+)').apply(pd.to_numeric)
clean_flats_data['room_count'] = clean_flats_data['room_count'].str.extract(r'(\d+)').apply(pd.to_numeric)

clean_flats_data['max_floor'] = clean_flats_data['floor'].str.split(' ').str[3].apply(pd.to_numeric)
clean_flats_data['floor'] = clean_flats_data['floor'].str.split(' ').str[1].apply(pd.to_numeric)

clean_flats_data['region'] = clean_flats_data['region'].str.replace('Київ', '').str.replace('р-н', '').str \
    .replace(" ", "").str.replace("\n", "")

clean_flats_data['street_number'] = clean_flats_data['street_name'].str.replace('Київ,', '').str.split(',').str[1].str.replace(" ", "") \
    .str.replace("\n", "")
clean_flats_data['street_name'] = clean_flats_data['street_name'].str.replace('Київ,', '').str.split(',').str[0].str \
    .replace("\n", "").str.strip()

clean_flats_data['living_square'] = clean_flats_data['square'].str.split('/').str[1].str.replace(" ", "").apply(
    pd.to_numeric)

clean_flats_data['kitchen_square'] = clean_flats_data['square'].str.split('/').str[2].str.replace("м²", "").str.replace(
    " ", "").apply(pd.to_numeric)

clean_flats_data['square'] = clean_flats_data['square'].str.split('/').str[0].str.replace(" ", "").apply(pd.to_numeric)
clean_flats_data = clean_flats_data.rename(columns={'square': 'total_square'})

# clean_flats_data['subway'] = clean_flats_data['subway'].apply(lambda x: 1 if pd.notnull(x) else 0)

clean_flats_data = clean_flats_data.dropna()
clean_flats_data = clean_flats_data.drop(clean_flats_data[clean_flats_data['total_square'] > 300].index)

clean_flats_data.to_excel("data/clean_flats_02_2024_db.xlsx")
