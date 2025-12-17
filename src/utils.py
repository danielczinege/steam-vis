# Mapping genres to indices - for the purpose of plotting heatmaps
genre_mapping = {
    'Casual' : 0, 
    'Racing': 1, 
    'Indie': 2, 
    'Sports': 3, 
    'RPG': 4, 
    'Simulation': 5, 
    'Strategy': 6, 
    'Action': 7, 
    'Adventure': 8, 
    'Action-Adventure': 9
    }
GENRES = ['Casual', 'Racing', 'Indie', 'Sports', 'RPG', 'Simulation', 'Strategy', 'Action', 'Adventure', 'Action-Adventure']
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
