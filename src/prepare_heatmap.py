## Creates a new .csv file that can be used directly to plot heatmap without further calculations.
from typing import List
import numpy as np
import pandas as pd
from utils import genre_mapping


# Read data
data = pd.read_csv("../data/games_processed.csv")
np_data = data.to_numpy()

# Indices for attributes
RELEASE_YEAR = 1
RELEASE_MONTH = 2
GENRES = 3

# Minimum for years, months to flatten to 0
all_years = set(np_data[:, RELEASE_YEAR])
years_min = min(all_years)

# List[genres][timeslot]
heatmap_data_year: List[List[int]] = [[0 for _ in range(max(all_years) - min(all_years) + 1)] for _ in range(10)]
heatmap_data_month: List[List[int]] = [[0 for _ in range(12)] for _ in range(10)]


for row in range(len(np_data)):
    genres_list: str = np_data[row][GENRES]
    year: int = np_data[row][RELEASE_YEAR]
    month: int = np_data[row][RELEASE_MONTH]
    
    # There are some instances of a game not having genres
    if isinstance(genres_list, float):
        continue
    
    for genre in genres_list.split(","):
        # print(year, years_min, year - years_min)
        heatmap_data_year[genre_mapping[genre]][year - years_min] += 1
        heatmap_data_month[genre_mapping[genre]][month - 1] += 1
        
    
np.savetxt("../data/games_per_year.txt", heatmap_data_year, delimiter=" ", fmt="%d")
np.savetxt("../data/games_per_months.txt", heatmap_data_month, delimiter=" ", fmt="%d")

