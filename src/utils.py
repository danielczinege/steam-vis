# Mapping genres to indices - for the purpose of plotting heatmaps
genre_mapping = {
    'Action' : 0, 
    'Adventure': 1, 
    'Strategy': 2, 
    'Casual': 3, 
    'Indie': 4, 
    'RPG': 5, 
    'Simulation': 6, 
    'Racing': 7, 
    'Sports': 8, 
    'Action-Adventure': 9
    }
GENRES = ['Action', 'Adventure', 'Strategy', 'Casual', 'Indie', 'RPG', 'Simulation','Racing', 'Sports', 'Action-Adventure']
# Ranges are (start, end] - inclusive from right
PRICE_RANGES = [(-1, 0), (0, 5), (5, 10), (10, 20), (20, 50)]
DLC_RANGES = [(-1, 0), (0, 1), (1, 2), (2, 5), (5, 10)]

PRICES_MAPPING = {
    0: "0",
    1: "(0,5]",
    2: "(5,10]",
    3: "(10,20]",
    4: "(20,50]",
    5: "50+"
}
DLCS_MAPPING = {
    0: "0",
    1: "1",
    2: "2",
    3: "(2,5]",
    4: "(5, 10]",
    5: "10+"
}
