import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

flats_data = pd.read_excel("data/clean_flats.xlsx")
print(flats_data.info())
sns.pairplot(flats_data,
             vars=['Region', 'Room count', 'Total Square', 'Floor', 'Subway', 'Max Floor', 'Living Square',
                   'Kitchen Square'], hue="Price")
data = flats_data[
    ['Region', 'Room count', 'Total Square', 'Floor', 'Subway', 'Max Floor', 'Living Square', 'Kitchen Square',
     'Price']]
sns.heatmap(data.corr(), annot=True)
plt.show()
