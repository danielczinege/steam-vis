import pandas as pd
from utils import genre_mapping, GENRES, PRICE_RANGES, DLC_RANGES, PRICES_MAPPING, DLCS_MAPPING

GENRES_POS = 3
PRICE_POS = 8
DLC_POS = 9
PCS_POS = -1  # last attribute


def load_dataset():
    df = pd.read_pickle('../data/games_processed.pkl')

    # To numpy array
    return df


def get_range(price: float, range):
    for i, (start, end) in enumerate(range):
        if start < price <= end:
            return i
    return len(range)


def dlcs_info(data, distribution: bool, quantity: int):
    count = quantity == 1
    col = PCS_POS if quantity == 0 else PRICE_POS
    
    # List for all (10) genres
    if count:
        quantity_per_dlc_r = {d_range: [0] for d_range in range(len(DLC_RANGES) + 1)}
    else:
        quantity_per_dlc_r = {d_range: [] for d_range in range(len(DLC_RANGES) + 1)}


    for row in range(len(data)):
        dlc_count: float = data[row][DLC_POS]
        dlc_range_i = get_range(dlc_count, DLC_RANGES)

        if not count:
            quantity_per_dlc_r[dlc_range_i].append(data[row][col])    
        else:
            quantity_per_dlc_r[dlc_range_i][0] += 1 

    if not distribution and not count:
        quantity_per_dlc_r = {DLCS_MAPPING[i]: sum(quantity_per_dlc_r[i]) / len(quantity_per_dlc_r[i]) 
                                for i in range(len(DLC_RANGES) + 1)}
    elif not distribution:
        quantity_per_dlc_r = {DLCS_MAPPING[i]: quantity_per_dlc_r[i] 
                                for i in range(len(DLC_RANGES) + 1)}
    else:
        print("dlcs distribution")
        for_dataframe = []
        for dlc_range in quantity_per_dlc_r.keys():
            for value in quantity_per_dlc_r[dlc_range]:
                for_dataframe.append([DLCS_MAPPING[dlc_range], value])
        return pd.DataFrame(for_dataframe, columns=["Number of Dlcs", 0])
        
    return pd.DataFrame.from_dict(quantity_per_dlc_r, orient="index")


def prices_info(data, distribution: bool, quantity: int, free: bool):
    print("in prices info")
    if quantity == 2:
        return pd.DataFrame([], columns=["Price ranges", 0])
    
    count = quantity == 1
    
    # List for all (10) genres
    if count:
        quantity_per_price_r = {price_r: [0] for price_r in range(len(PRICE_RANGES) + 1)}
    else:
        quantity_per_price_r = {price_r: [] for price_r in range(len(PRICE_RANGES) + 1)}

    print("processing price")

    for row in range(len(data)):
        price: float = data[row][PRICE_POS]
        price_range_i = get_range(price, PRICE_RANGES)

        if count:
            quantity_per_price_r[price_range_i][0] += 1
        else:
            quantity_per_price_r[price_range_i].append(data[row][PCS_POS])

    start = 0 if not free else 1
    if not distribution and not count:
        print("price one")
        quantity_per_price_r = {PRICES_MAPPING[i]: sum(quantity_per_price_r[i]) / len(quantity_per_price_r[i]) 
                                for i in range(start, len(PRICE_RANGES) + 1)}
    elif not distribution:
        print("price not dist")
        quantity_per_price_r = {PRICES_MAPPING[i]: quantity_per_price_r[i] 
                                for i in range(start, len(PRICE_RANGES) + 1)}
    else:
        print("price distribution")
        for_dataframe = []
        for price_range in quantity_per_price_r.keys():
            for value in quantity_per_price_r[price_range]:
                for_dataframe.append([PRICES_MAPPING[price_range], value])
        return pd.DataFrame(for_dataframe, columns=["Price ranges", 0])
        
    return pd.DataFrame.from_dict(quantity_per_price_r, orient="index")


# Quantity: 0 - pcs_pos, 1 - count, 2 - price
def genres_info(data, distribution: bool, quantity: int):
    count = quantity == 1
    col = PCS_POS if quantity == 0 else PRICE_POS
        
    # List for all (10) genres
    if count:
        quantity_per_genre = {genre: [0] for genre in GENRES}
    else:
        quantity_per_genre = {genre: [] for genre in GENRES}
                
    for row in range(len(data)):
        genres_list: str = data[row][GENRES_POS]
        
        # There are some instances of a game not having genres
        if isinstance(genres_list, float):
            continue
        
        if not count:
            for genre in genres_list:
                quantity_per_genre[genre].append(data[row][col])    
        else:
            for genre in genres_list:
                quantity_per_genre[genre][0] += 1   

    if not distribution and not count:
        quantity_per_genre = {genre: sum(quantity_per_genre[genre]) / len(quantity_per_genre[genre]) for genre in GENRES}

    if distribution:
        for_dataframe = []
        for genre in quantity_per_genre.keys():
            for value in quantity_per_genre[genre]:
                for_dataframe.append([genre, value])
        return pd.DataFrame(for_dataframe, columns=["Genre", 0])
        
    return pd.DataFrame.from_dict(quantity_per_genre, orient="index")


# Distribution - True for distribution, False for average
# Quantity - 0: review, 1: count, 2: price
# Cat - 0: genres, 1: price, 2: dlcs
def l2_get_data(df, distribution: bool, quantity: int, free: bool):
    if free:
        df = df[df[:, PRICE_POS] != 0]
    if quantity == 0:
        df = df[df[:, PCS_POS] != -1]
        
    (g_data, p_data, d_data) = (genres_info(df, distribution, quantity), 
                                prices_info(df, distribution, quantity, free), 
                                dlcs_info(df, distribution, quantity))
    
    data_name: str = "Count" if quantity == 1 else ("% of positive reviews" if quantity == 0 else "Price")
    
    g_data = g_data.rename(columns={0: data_name})
    p_data = p_data.rename(columns={0: data_name})
    d_data = d_data.rename(columns={0: data_name})
    return g_data, p_data, d_data
