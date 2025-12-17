import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset (example CSV)
df = pd.read_csv("./data/games_03_2025_reduced.csv")

print("="*80)
print("EXPLORING AND REMOVING DUPLICATES")
print("="*80)

# --- 1. Explore and remove duplicates ---
# Check for duplicates based on 'name' and 'release_date'
duplicates = df.duplicated(subset=['name', 'release_date'], keep=False)
duplicate_count = duplicates.sum()

print(f"Found {duplicate_count} potential duplicates based on 'name' and 'release_date'.")

# Remove duplicates
df.drop_duplicates(subset=['name', 'release_date'], inplace=True)
print(f"Removed duplicates. New dataset size: {df.shape[0]:,} rows")


print("\n" + "="*80)
print("COMPREHENSIVE MISSING VALUES ANALYSIS")
print("="*80)

# --- 2. Parse JSON/array columns and analyze missing values ---
print(f"Original dataset size: {df.shape[0]:,} rows, {df.shape[1]} columns")

def parse_tags(x):
    """
    Parses a string representation of a dictionary or an empty list.
    Handles NaN values, dictionary-like strings ("{'tag': count}"),
    and empty list strings ("[]").
    """
    if pd.isna(x) or x == '[]':
        return {}
    try:
        return ast.literal_eval(str(x))
    except (ValueError, SyntaxError):
        return {}

def parse_genres(x):
    """Parse genres from various formats"""
    if pd.isna(x):
        return []
    if isinstance(x, str):
        if x.startswith('[') and x.endswith(']'):
            try:
                return ast.literal_eval(x)
            except:
                # If it's just a comma-separated string
                return [genre.strip().strip('"').strip("'") for genre in x.strip('[]').split(',') if genre.strip()]
        elif ',' in x:
            return [genre.strip() for genre in x.split(',') if genre.strip()]
        else:
            return [x.strip()] if x.strip() else []
    elif isinstance(x, list):
        return x
    else:
        return []

def parse_categories(x):
    """Parse categories using the same logic as genres"""
    return parse_genres(x)

def parse_supported_languages(x):
    """Parse supported languages from various formats"""
    if pd.isna(x):
        return []
    if isinstance(x, str):
        if x.startswith('[') and x.endswith(']'):
            try:
                return ast.literal_eval(x)
            except:
                # If it's just a comma-separated string
                return [lang.strip().strip('"').strip("'") for lang in x.strip('[]').split(',') if lang.strip()]
        elif ',' in x:
            return [lang.strip() for lang in x.split(',') if lang.strip()]
        else:
            return [x.strip()] if x.strip() else []
    elif isinstance(x, list):
        return x
    else:
        return []

def parse_estimated_owners(x):
    """Parse estimated owners - clean and validate the range format"""
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        if x == '' or x.lower() == 'unknown':
            return None
        # Expected format: "50000000 - 100000000" or similar
        if ' - ' in x:
            return x
        # Handle other potential formats
        elif '-' in x:
            return x.replace('-', ' - ')
        else:
            return x if x else None
    return str(x) if x else None

# Parse all columns
df['supported_languages'] = df['supported_languages'].apply(parse_supported_languages)
df['tags'] = df['tags'].apply(parse_tags)
df['genres_parsed'] = df['genres'].apply(parse_genres)
df['categories_parsed'] = df['categories'].apply(parse_categories)
df['estimated_owners_parsed'] = df['estimated_owners'].apply(parse_estimated_owners)

print("\n" + "="*60)
print("DETAILED MISSING VALUES ANALYSIS BY COLUMN")
print("="*60)

# --- 3. Analyze ESTIMATED_OWNERS ---
print(f"\n--- ESTIMATED_OWNERS ANALYSIS ---")
original_nulls = df['estimated_owners'].isnull().sum()
print(f"Original NULL values: {original_nulls:,} ({original_nulls/len(df)*100:.1f}%)")

empty_strings = (df['estimated_owners'] == '').sum()
print(f"Empty strings: {empty_strings:,} ({empty_strings/len(df)*100:.1f}%)")

parsed_nulls = df['estimated_owners_parsed'].isnull().sum()
print(f"NULL after parsing: {parsed_nulls:,} ({parsed_nulls/len(df)*100:.1f}%)")

valid_owners = df['estimated_owners_parsed'].notnull().sum()
print(f"Valid estimated_owners: {valid_owners:,} ({valid_owners/len(df)*100:.1f}%)")

# Show unique patterns in estimated_owners
if valid_owners > 0:
    print(f"\nUnique estimated_owners patterns (top 10):")
    owners_counts = df['estimated_owners_parsed'].value_counts().head(10)
    for pattern, count in owners_counts.items():
        print(f"  '{pattern}': {count:,}")

# --- 4. Analyze TAGS (BEFORE FILTERING) ---
print(f"\n--- TAGS ANALYSIS (BEFORE FILTERING) ---")
original_nulls = df['tags'].apply(lambda x: pd.isna(x) if not isinstance(x, dict) else False).sum()
print(f"Original NULL values: {original_nulls:,} ({original_nulls/len(df)*100:.1f}%)")

empty_dicts = df['tags'].apply(lambda x: len(x) == 0 if isinstance(x, dict) else True).sum()
print(f"Empty tag dictionaries after parsing: {empty_dicts:,} ({empty_dicts/len(df)*100:.1f}%)")

games_with_tags = df['tags'].apply(lambda x: len(x) > 0 if isinstance(x, dict) else False).sum()
print(f"Games with at least one tag: {games_with_tags:,} ({games_with_tags/len(df)*100:.1f}%)")

if games_with_tags > 0:
    # Count all tags
    all_tags = []
    for tag_dict in df['tags']:
        if isinstance(tag_dict, dict):
            all_tags.extend(tag_dict.keys())

    if all_tags:
        tag_counts = Counter(all_tags)
        print(f"\nTop 10 most common tags:")
        for tag, count in tag_counts.most_common(10):
            print(f"  '{tag}': {count:,}")

        avg_tags_per_game = df['tags'].apply(lambda x: len(x) if isinstance(x, dict) else 0).mean()
        print(f"\nAverage tags per game (before filtering): {avg_tags_per_game:.2f}")

# --- NEW: FILTER TAGS BY THRESHOLD ---
print("\n" + "="*60)
print("FILTERING TAGS BY VOTE THRESHOLD")
print("="*60)

def filter_tags_by_threshold(tags_dict, threshold_percentile=0.15):
    """
    Filter tags to keep only those with vote counts above a dynamic threshold.
    
    Args:
        tags_dict: Dictionary of {tag: vote_count}
        threshold_percentile: Keep tags with votes >= this percentile of the max votes
                              (0.15 = 15% of the highest vote count)
    
    Returns:
        Filtered dictionary with only relevant tags
    """
    if not isinstance(tags_dict, dict) or len(tags_dict) == 0:
        return {}
    
    # Get the maximum vote count for this game
    max_votes = max(tags_dict.values())
    
    # Calculate threshold (e.g., 15% of the max votes)
    threshold = max_votes * threshold_percentile
    
    # Keep tags with votes >= threshold
    filtered = {tag: count for tag, count in tags_dict.items() if count >= threshold}
    
    return filtered

# Apply tag filtering
print(f"Applying tag filtering with 15% threshold (keeps tags with votes >= 15% of max votes)...")
df['tags_filtered'] = df['tags'].apply(filter_tags_by_threshold)

# Analyze filtered tags
print(f"\n--- TAGS ANALYSIS (AFTER FILTERING) ---")
games_with_tags_filtered = df['tags_filtered'].apply(lambda x: len(x) > 0 if isinstance(x, dict) else False).sum()
games_with_just_one_tag = df['tags_filtered'].apply(lambda x: len(x) == 1 if isinstance(x, dict) else False).sum()
print(f"Games with at least one tag after filtering: {games_with_tags_filtered:,} ({games_with_tags_filtered/len(df)*100:.1f}%)")
print(f"Games with just one tag after filtering: {games_with_just_one_tag:,} ({games_with_just_one_tag/len(df)*100:.1f}%)")

if games_with_tags_filtered > 0:
    avg_tags_per_game_filtered = df['tags_filtered'].apply(lambda x: len(x) if isinstance(x, dict) else 0).mean()
    print(f"Average tags per game (after filtering): {avg_tags_per_game_filtered:.2f}")
    
    # Count reduction
    total_tags_before = df['tags'].apply(lambda x: len(x) if isinstance(x, dict) else 0).sum()
    total_tags_after = df['tags_filtered'].apply(lambda x: len(x) if isinstance(x, dict) else 0).sum()
    reduction_pct = ((total_tags_before - total_tags_after) / total_tags_before * 100) if total_tags_before > 0 else 0
    print(f"\nTotal tags reduced from {total_tags_before:,} to {total_tags_after:,} ({reduction_pct:.1f}% reduction)")

# Replace original tags with filtered tags
df['tags'] = df['tags_filtered']
df = df.drop(columns=['tags_filtered'])

# --- 5. Analyze NAME ---
print(f"\n--- NAME ANALYSIS ---")
original_nulls = df['name'].isnull().sum()
print(f"Original NULL values: {original_nulls:,} ({original_nulls/len(df)*100:.1f}%)")

empty_strings = (df['name'] == '').sum()
print(f"Empty strings: {empty_strings:,} ({empty_strings/len(df)*100:.1f}%)")

whitespace_only = df['name'].apply(lambda x: isinstance(x, str) and x.strip() == '').sum()
print(f"Whitespace-only strings: {whitespace_only:,} ({whitespace_only/len(df)*100:.1f}%)")

total_missing_names = original_nulls + empty_strings + whitespace_only
print(f"Total missing/empty names: {total_missing_names:,} ({total_missing_names/len(df)*100:.1f}%)")

valid_names = len(df) - total_missing_names
print(f"Valid names: {valid_names:,} ({valid_names/len(df)*100:.1f}%)")

# Show some examples of problematic names if any
if total_missing_names > 0:
    print(f"\nSample of games with missing/empty names:")
    problem_mask = df['name'].isnull() | (df['name'] == '') | df['name'].apply(lambda x: isinstance(x, str) and x.strip() == '')
    problem_sample = df[problem_mask][['appid', 'name']].head(5)
    for idx, row in problem_sample.iterrows():
        print(f"  AppID {row['appid']}: name='{row['name']}'")

# --- 6. Analyze CATEGORIES ---
print(f"\n--- CATEGORIES ANALYSIS ---")
original_nulls = df['categories'].isnull().sum()
print(f"Original NULL values in categories: {original_nulls:,} ({original_nulls/len(df)*100:.1f}%)")

empty_arrays = df['categories_parsed'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()
print(f"Empty category arrays after parsing: {empty_arrays:,} ({empty_arrays/len(df)*100:.1f}%)")

games_with_categories = df['categories_parsed'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
print(f"Games with at least one category: {games_with_categories:,} ({games_with_categories/len(df)*100:.1f}%)")

if games_with_categories > 0:
    # Count all categories
    all_categories = []
    for categories_list in df['categories_parsed']:
        if isinstance(categories_list, list):
            all_categories.extend(categories_list)

    if all_categories:
        category_counts = Counter(all_categories)
        print(f"\nTop 10 most common categories:")
        for category, count in category_counts.most_common(10):
            print(f"  '{category}': {count:,}")

        avg_categories_per_game = df['categories_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        print(f"\nAverage categories per game: {avg_categories_per_game:.2f}")

# --- 7. Analyze SUPPORTED_LANGUAGES ---
print(f"\n--- SUPPORTED_LANGUAGES ANALYSIS ---")
original_nulls = df['supported_languages'].apply(lambda x: pd.isna(x) if not isinstance(x, list) else False).sum()
print(f"Original NULL values: {original_nulls:,} ({original_nulls/len(df)*100:.1f}%)")

empty_arrays = df['supported_languages'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()
print(f"Empty language arrays after parsing: {empty_arrays:,} ({empty_arrays/len(df)*100:.1f}%)")

games_with_languages = df['supported_languages'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
print(f"Games with at least one supported language: {games_with_languages:,} ({games_with_languages/len(df)*100:.1f}%)")

if games_with_languages > 0:
    # Count all languages
    all_languages = []
    for languages_list in df['supported_languages']:
        if isinstance(languages_list, list):
            all_languages.extend(languages_list)

    if all_languages:
        language_counts = Counter(all_languages)
        print(f"\nTop 10 most supported languages:")
        for language, count in language_counts.most_common(10):
            print(f"  '{language}': {count:,}")

        avg_languages_per_game = df['supported_languages'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        print(f"\nAverage supported languages per game: {avg_languages_per_game:.2f}")

# --- 8. GENRES ANALYSIS ---
print(f"\n--- GENRES ANALYSIS ---")
original_nulls = df['genres'].isnull().sum()
print(f"Original NULL values in genres: {original_nulls:,} ({original_nulls/len(df)*100:.1f}%)")

empty_arrays = df['genres_parsed'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()
print(f"Empty genre arrays after parsing: {empty_arrays:,} ({empty_arrays/len(df)*100:.1f}%)")

games_with_genres = df['genres_parsed'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
print(f"Games with at least one genre: {games_with_genres:,} ({games_with_genres/len(df)*100:.1f}%)")

if games_with_genres > 0:
    # Count all genres
    all_genres = []
    for genres_list in df['genres_parsed']:
        if isinstance(genres_list, list):
            all_genres.extend(genres_list)

    if all_genres:
        genre_counts = Counter(all_genres)
        print(f"\nTop 10 most common genres:")
        for genre, count in genre_counts.most_common(10):
            print(f"  '{genre}': {count:,}")

        avg_genres_per_game = df['genres_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        print(f"\nAverage genres per game: {avg_genres_per_game:.2f}")

# --- 9. Pseudo-missing values analysis (zeros in numeric columns) ---
print("\n--- Potential pseudo-missing values (zeros in numeric columns) ---")
num_cols = df.select_dtypes(include=[np.number]).columns
zero_counts = (df[num_cols] == 0).sum().sort_values(ascending=False)
print(zero_counts)

print("\n" + "-"*60)
print("Cleaning price column... (using tags to determine free-to-play games)")
print("-"*60)

# --- 10. Handle price=0 ---
def clean_price(row):
    """
    Cleans the price column. If price is 0, checks if 'Free to Play' is
    present in either tags or genres. If not, the price is set to NaN.
    """
    price = row['price']
    tags = row['tags']
    genres = row['genres_parsed']

    if price == 0:
        # Check if 'Free to Play' is in tags (case-insensitive)
        is_free_in_tags = isinstance(tags, dict) and any(t.lower() == "free to play" for t in tags.keys())

        # Check if 'Free to Play' is in genres (case-insensitive)
        is_free_in_genres = isinstance(genres, list) and any(g.lower() == "free to play" for g in genres)

        # If it's marked as Free to Play in either tags or genres, keep the price at 0
        if is_free_in_tags or is_free_in_genres:
            return 0  # It's a genuinely free-to-play game
        else:
            # If price is 0 but not marked as F2P, treat it as missing data
            return np.nan

    return price

# Overwrite original price column with cleaned data
df['price'] = df.apply(clean_price, axis=1)


# --- 11. REMOVE ROWS WITH NaN VALUES IN KEY COLUMNS ---
print("\n" + "="*60)
print("FILTERING DATASET FOR COMPLETE KEY COLUMNS")
print("="*60)

# Key columns we need for analysis
key_columns = ['price']

print(f"\nMissing values in key columns before filtering:")
for col in key_columns:
    missing_count = df[col].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"  {col}: {missing_count:,} ({missing_pct:.1f}%)")

# Remove rows that have NaN in any of the key columns
df_complete = df.dropna(subset=key_columns)
print(f"\nFiltered dataset size: {df_complete.shape[0]:,} rows")
print(f"Data retention: {len(df_complete)/len(df)*100:.1f}%")


# --- 12. ADDITIONAL FILTERING: Remove games with empty genres, categories, and supported languages ---
print("\n" + "="*60)
print("ADDITIONAL FILTERING: REMOVE EMPTY GENRES, CATEGORIES, LANGUAGES")
print("="*60)

print(f"Starting with: {df_complete.shape[0]:,} rows")

# Create masks for additional filtering criteria
empty_genres_mask = df_complete['genres_parsed'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
empty_categories_mask = df_complete['categories_parsed'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
empty_languages_mask = df_complete['supported_languages'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)

print(f"\nGames to be removed:")
print(f"  Empty genres: {empty_genres_mask.sum():,} ({empty_genres_mask.mean()*100:.1f}%)")
print(f"  Empty categories: {empty_categories_mask.sum():,} ({empty_categories_mask.mean()*100:.1f}%)")
print(f"  Empty supported languages: {empty_languages_mask.sum():,} ({empty_languages_mask.mean()*100:.1f}%)")

# Combine all criteria - remove rows where ANY of these are empty
combined_mask = empty_genres_mask | empty_categories_mask | empty_languages_mask
games_to_remove = combined_mask.sum()
print(f"  Total unique games to remove: {games_to_remove:,} ({games_to_remove/len(df_complete)*100:.1f}%)")

# Apply additional filtering
df_fully_complete = df_complete[~combined_mask]
print(f"\nFully filtered dataset size: {df_fully_complete.shape[0]:,} rows")
print(f"Data retention from original: {len(df_fully_complete)/len(df)*100:.1f}%")
print(f"Data retention from key-columns-filtered: {len(df_fully_complete)/len(df_complete)*100:.1f}%")

# --- FINAL MISSING VALUES SUMMARY ---
print("\n" + "="*40)
print("FINAL MISSING VALUES SUMMARY (FULLY FILTERED DATASET)")
print("="*40)

print(f"Final dataset size: {df_fully_complete.shape[0]:,} rows")

# Standard missing values in fully filtered dataset
print(f"\n--- Standard Missing Values (NaN) in Fully Filtered Dataset ---")
basic_missing_final = df_fully_complete.isnull().sum()
any_missing = False
for col in df_fully_complete.columns:
    missing_count = basic_missing_final[col]
    if len(df_fully_complete) > 0:
        missing_pct = (missing_count / len(df_fully_complete)) * 100
    else:
        missing_pct = 0
    if missing_count > 0:  # Only show columns with missing values
        print(f"{col:30}: {missing_count:8,} ({missing_pct:5.1f}%)")
        any_missing = True

if not any_missing:
    print("No standard missing values (NaN) in fully filtered dataset!")

print(f'\ntags (empty dict): {df_fully_complete["tags"].apply(lambda x: len(x) == 0 if isinstance(x, dict) else True).sum()}')

# Check that our filtering worked
print(f"\n--- Verification of Filtering ---")
print(f"Games with empty genres: {df_fully_complete['genres_parsed'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()}")
print(f"Games with empty categories: {df_fully_complete['categories_parsed'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()}")
print(f"Games with empty supported languages: {df_fully_complete['supported_languages'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()}")
print(f"Games with NaN price: {df_fully_complete['price'].isnull().sum()}")


# --- 13. ANALYSIS ON FULLY FILTERED DATASET ---
print("\n" + "="*60)
print("ANALYSIS ON FULLY FILTERED DATASET")
print("="*60)

if len(df_fully_complete) > 0:
    # Analysis for average_playtime_forever
    playtime_hours = df_fully_complete['average_playtime_forever'] / 60

    print(f"\n--- Average Playtime Forever Analysis (n={len(df_fully_complete):,}) ---")
    print(f"Mean: {playtime_hours.mean():.2f} hours")
    print(f"Median: {playtime_hours.median():.2f} hours")
    print(f"Standard Deviation: {playtime_hours.std():.2f} hours")
    print(f"95th percentile: {playtime_hours.quantile(0.95):.2f} hours")
    print(f"99th percentile: {playtime_hours.quantile(0.99):.2f} hours")

    # Price analysis
    price_data = df_fully_complete['price']
    print(f"\n--- Price Analysis (n={len(df_fully_complete):,}) ---")
    print(f"Mean: ${price_data.mean():.2f}")
    print(f"Median: ${price_data.median():.2f}")

    free_games = (price_data == 0).sum()
    paid_games = (price_data > 0).sum()
    print(f"\nFree games: {free_games:,} ({free_games/len(df_fully_complete)*100:.1f}%)")
    print(f"Paid games: {paid_games:,} ({paid_games/len(df_fully_complete)*100:.1f}%)")

    # Additional stats on fully filtered dataset
    print(f"\n--- Additional Statistics (Fully Filtered Dataset) ---")
    print(f"Average genres per game: {df_fully_complete['genres_parsed'].apply(len).mean():.2f}")
    print(f"Average categories per game: {df_fully_complete['categories_parsed'].apply(len).mean():.2f}")
    print(f"Average supported languages per game: {df_fully_complete['supported_languages'].apply(len).mean():.2f}")
    print(f"Average tags per game: {df_fully_complete['tags'].apply(len).mean():.2f}")

    print("\n--- Potential pseudo-missing values (zeros in numeric columns) ---")
    num_cols = df_fully_complete.select_dtypes(include=[np.number]).columns
    zero_counts = (df_fully_complete[num_cols] == 0).sum().sort_values(ascending=False)
    print(zero_counts)

    # --- 14. Visualization of Missing Values ---
    fig, ((ax2, ax4)) = plt.subplots(1, 2, figsize=(15, 6))

    # Distribution of prices (fully filtered dataset)
    df_fully_complete['price'].plot(kind='hist', bins=50, logy=True, title="Price Distribution (Fully Filtered Dataset)", ax=ax2)
    ax2.set_xlabel("Price ($)")
    ax2.set_ylabel("Count (log scale)")

    # Data retention visualization
    retention_data = {
        'Original': len(df),
        'Key Columns\nFiltered': len(df_complete),
        'Fully Filtered\n(+genres/categories/languages)': len(df_fully_complete)
    }

    bars = ax4.bar(retention_data.keys(), retention_data.values(),
                   color=['lightcoral', 'lightskyblue', 'lightgreen'])
    ax4.set_title("Dataset Size After Filtering Steps")
    ax4.set_ylabel("Number of Games")
    ax4.tick_params(axis='x', rotation=45)

    # Add percentage labels on bars
    for i, (key, value) in enumerate(retention_data.items()):
        if i == 0:
            pct = 100.0
        else:
            pct = (value / len(df)) * 100
        ax4.text(i, value + len(df)*0.01, f'{value:,}\n({pct:.1f}%)',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

else:
    print("No data left after filtering for analysis.")


print("\nAnalysis complete! Dataset columns available:")
print(list(df_fully_complete.columns))

# print()
# unique_genres = np.unique(np.concatenate(df_fully_complete["genres_parsed"].values))
# print(unique_genres)

# print()
# unique_categories = np.unique(np.concatenate(df_fully_complete["categories_parsed"].values))
# print(unique_categories)

# print()
# unique_tags = np.unique(np.concatenate(df_fully_complete["tags"].apply(lambda x : list(x.keys())).values))
# print(unique_tags)

STRICT_GENRE_MAP = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Casual": "Casual",
    "Indie": "Indie",
    "RPG": "RPG",
    "Racing": "Racing",
    "Simulation": "Simulation",
    "Sports": "Sports",
    "Strategy": "Strategy",
}

STRICT_TAG_MAP = {
    # ACTION
    'Action': 'Action',
    'Arena Shooter': 'Action',
    "Beat 'em up": 'Action',
    'Boomer Shooter': 'Action',
    'Boss Rush': 'Action',
    'Bullet Hell': 'Action',
    'Extraction Shooter': 'Action',
    'FPS': 'Action',
    'Fighting': 'Action',
    'Hack and Slash': 'Action',
    'Hero Shooter': 'Action',
    'Looter Shooter': 'Action',
    'Musou': 'Action',
    'On-Rails Shooter': 'Action',
    "Shoot 'Em Up": 'Action',
    'Shooter': 'Action',
    'Side Scroller': 'Action',
    'Third-Person Shooter': 'Action',
    'Top-Down Shooter': 'Action',
    'Twin Stick Shooter': 'Action',

    # ACTION-ADVENTURE
    'Action-Adventure': 'Action-Adventure',
    'Dungeon Crawler': 'Action-Adventure',
    'Metroidvania': 'Action-Adventure',
    'Stealth': 'Action-Adventure',

    # ADVENTURE
    'Adventure': 'Adventure',
    'Choose Your Own Adventure': 'Adventure',
    'Point & Click': 'Adventure',
    'FMV': 'Adventure',
    'Interactive Fiction': 'Adventure',
    'Visual Novel': 'Adventure',
    'Walking Simulator': 'Adventure',
    'Hidden Object': 'Adventure',

    # CASUAL
    'Casual': 'Casual',
    'Clicker': 'Casual',
    'Idler': 'Casual',
    'Match 3': 'Casual',
    'Mini Golf': 'Casual',
    'Minigames': 'Casual',
    'Party Game': 'Casual',
    'Solitaire': 'Casual',
    'Sokoban': 'Casual',
    'Word Game': 'Casual',

    # RPG
    'RPG': 'RPG',
    'CRPG': 'RPG',
    'JRPG': 'RPG',
    'RPGMaker': 'RPG',
    'Party-Based RPG': 'RPG',
    'Tactical RPG': 'RPG',
    'Strategy RPG': 'RPG',
    'MMORPG': 'RPG',

    # STRATEGY
    '4X': 'Strategy',
    'Auto Battler': 'Strategy',
    'City Builder': 'Strategy',
    'Grand Strategy': 'Strategy',
    'RTS': 'Strategy',
    'Tactical': 'Strategy',
    'Tower Defense': 'Strategy',
    'Turn-Based Strategy': 'Strategy',
    'Turn-Based Tactics': 'Strategy',
    'Wargame': 'Strategy',

    # SIMULATION
    'Simulation': 'Simulation',
    'Automobile Sim': 'Simulation',
    'Farming Sim': 'Simulation',
    'Flight': 'Simulation',
    'Hobby Sim': 'Simulation',
    'Job Simulator': 'Simulation',
    'Life Sim': 'Simulation',
    'Medical Sim': 'Simulation',
    'Space Sim': 'Simulation',
    'Trains': 'Simulation',

    # INDIE
    'Indie': 'Indie',
    'GameMaker': 'Indie',
    'RPGMaker': 'Indie',
    'Crowdfunded': 'Indie',
    'Kickstarter': 'Indie',

    # SPORTS
    'Baseball': 'Sports',
    'Basketball': 'Sports',
    'Bowling': 'Sports',
    'Boxing': 'Sports',
    'Cricket': 'Sports',
    'Football': 'Sports',
    'Football (American)': 'Sports',
    'Football (Soccer)': 'Sports',
    'Golf': 'Sports',
    'Hockey': 'Sports',
    'Skateboarding': 'Sports',
    'Skiing': 'Sports',
    'Snowboarding': 'Sports',
    'Soccer': 'Sports',
    'Tennis': 'Sports',
    'Volleyball': 'Sports',
    'Wrestling': 'Sports',

    # RACING
    'Racing': 'Racing',
    'Motorbike': 'Racing',
    'Motocross': 'Racing',
    'ATV': 'Racing',
    'Offroad': 'Racing',
    'Driving': 'Racing',
}

def map_strict_genres(genres):
    result = set()
    for g in genres:
        if g in STRICT_GENRE_MAP:
            result.add(STRICT_GENRE_MAP[g])
    return result

def map_strict_tags(tags_dict):
    result = set()
    for tag in tags_dict.keys():
        if tag in STRICT_TAG_MAP:
            result.add(STRICT_TAG_MAP[tag])
    return result

df_fully_complete = df_fully_complete.copy()

df_fully_complete["strict_genre_categories"] = df_fully_complete["genres_parsed"].apply(map_strict_genres)
df_fully_complete["strict_tag_categories"] = df_fully_complete["tags"].apply(map_strict_tags)

df_fully_complete["strict_combined_categories"] = df_fully_complete.apply(
    lambda row: row["strict_genre_categories"] | row["strict_tag_categories"], 
    axis=1
)

games_with_genres = df_fully_complete['strict_combined_categories'].apply(lambda x: len(x) > 0).sum()
print(f"Games with at least one genre: {games_with_genres:,} ({games_with_genres/len(df_fully_complete)*100:.1f}%)")

# --- 15. PREPARE FINAL DATASET WITH SELECTED COLUMNS ---
print("\n" + "="*60)
print("PREPARING FINAL DATASET")
print("="*60)

# Extract year and month from release_date
df_fully_complete['release_year'] = pd.to_datetime(df_fully_complete['release_date'], errors='coerce').dt.year
df_fully_complete['release_month'] = pd.to_datetime(df_fully_complete['release_date'], errors='coerce').dt.month

# Select and rename columns
df_final = df_fully_complete[[
    'name',
    'release_year',
    'release_month',
    'strict_combined_categories',  # Will be renamed to 'genres'
    'supported_languages',
    'windows',
    'mac',
    'linux',
    'price',
    'dlc_count',
    'pct_pos_total'
]].copy()

# Rename strict_combined_categories to genres
df_final.rename(columns={'strict_combined_categories': 'genres'}, inplace=True)

# Remove rows with invalid release dates (NaT converted to NaN)
print(f"\nRows before removing invalid dates: {len(df_final):,}")
df_final = df_final.dropna(subset=['release_year', 'release_month'])
df_final['release_year'] = df_final['release_year'].astype(int)
df_final['release_month'] = df_final['release_month'].astype(int)
print(f"Rows after removing invalid dates: {len(df_final):,}")

# Check pct_pos_total values (but don't remove -1)
print(f"\nAnalyzing pct_pos_total values:")
invalid_pct = (df_final['pct_pos_total'] == -1).sum()
valid_pct = (df_final['pct_pos_total'] != -1).sum()
print(f"  Valid values (not -1): {valid_pct:,} ({valid_pct/len(df_final)*100:.1f}%)")
print(f"  Invalid values (-1): {invalid_pct:,} ({invalid_pct/len(df_final)*100:.1f}%)")
print(f"  Note: Keep all rows; filter pct_pos_total != -1 only when needed for specific charts")

# Display final dataset info
print(f"\nFinal dataset shape: {df_final.shape}")
print(f"\nColumn data types:")
print(df_final.dtypes)

print(f"\nFirst few rows:")
print(df_final.head())

print(f"\nDataset statistics:")
print(df_final.describe())

print(f"\nMissing values in final dataset:")
missing_final = df_final.isnull().sum()
if missing_final.sum() > 0:
    print(missing_final[missing_final > 0])
else:
    print("No missing values!")

# --- 16. SAVE TO FILE ---
output_file = "./data/games_processed.pkl"
print(f"\nSaving processed dataset to: {output_file}")
df_final.to_pickle(output_file)
print("✓ Dataset saved successfully!")

# Also save as CSV for easy inspection (genres will be converted to string representation)
csv_output = "./data/games_processed.csv"
df_csv = df_final.copy()
df_csv['genres'] = df_csv['genres'].apply(lambda x: ','.join(sorted(x)) if x else '')
df_csv['supported_languages'] = df_csv['supported_languages'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
df_csv.to_csv(csv_output, index=False)
print(f"✓ CSV version saved to: {csv_output}")

print(f"\n{'='*60}")
print("PROCESSING COMPLETE!")
print(f"{'='*60}")
print(f"Final dataset: {len(df_final):,} games")
print(f"Retention from original: {len(df_final)/len(df)*100:.1f}%")
print(f"\nTo load the data in another script, use:")
print(f"  import pandas as pd")
print(f"  df = pd.read_pickle('{output_file}')")
print(f"\nWhen using pct_pos_total for charts, filter with:")
print(f"  df_with_reviews = df[df['pct_pos_total'] != -1]")
