#!/usr/bin/env python3
"""
Real Large Text Classification Dataset Downloader
Downloads actual datasets from multiple sources to reach 300MB+ target
NO SYNTHETIC DATA - Only real datasets from internet
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import requests
from sklearn.utils import shuffle
import urllib.request
import json


def check_kaggle_setup():
    """Check if Kaggle API is properly set up"""
    try:
        import kaggle
        kaggle.api.authenticate()
        print("✅ Kaggle API is properly configured")
        return True
    except ImportError:
        print("❌ Kaggle package not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        return check_kaggle_setup()
    except Exception as e:
        print(f"❌ Kaggle API not configured: {e}")
        return False


def download_20newsgroups():
    """Download 20 Newsgroups dataset (18K+ documents)"""
    try:
        from sklearn.datasets import fetch_20newsgroups
        print("📥 Downloading 20 Newsgroups dataset...")

        newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

        all_data = list(newsgroups_train.data) + list(newsgroups_test.data)
        all_targets = list(newsgroups_train.target) + list(newsgroups_test.target)
        all_target_names = newsgroups_train.target_names

        df = pd.DataFrame({
            'text': all_data,
            'category': [all_target_names[target] for target in all_targets]
        })

        df = df[df['text'].str.len() > 50]  # Remove very short texts
        print(f"✅ 20 Newsgroups: {len(df)} samples, Size: ~{len(df) * 500 / 1024 / 1024:.1f}MB")
        return df

    except Exception as e:
        print(f"❌ Failed to download 20 Newsgroups: {e}")
        return None


def download_imdb_reviews():
    """Download IMDB Movie Reviews dataset (50K reviews)"""
    try:
        print("📥 Downloading IMDB Movie Reviews dataset...")

        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        # Download and extract
        os.makedirs("temp_data", exist_ok=True)

        print("   Downloading IMDB dataset (84MB)...")
        response = requests.get(url, stream=True)
        with open("temp_data/imdb.tar.gz", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("   Extracting dataset...")
        import tarfile
        with tarfile.open("temp_data/imdb.tar.gz", "r:gz") as tar:
            tar.extractall("temp_data/")

        # Read positive and negative reviews
        data = []

        # Training data
        for sentiment in ['pos', 'neg']:
            folder = f"temp_data/aclImdb/train/{sentiment}"
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.endswith('.txt'):
                        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                            text = f.read()
                            data.append({'text': text, 'category': 'positive' if sentiment == 'pos' else 'negative'})

        # Test data
        for sentiment in ['pos', 'neg']:
            folder = f"temp_data/aclImdb/test/{sentiment}"
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.endswith('.txt'):
                        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                            text = f.read()
                            data.append({'text': text, 'category': 'positive' if sentiment == 'pos' else 'negative'})

        df = pd.DataFrame(data)
        print(f"✅ IMDB Reviews: {len(df)} samples, Size: ~{len(df) * 1000 / 1024 / 1024:.1f}MB")
        return df

    except Exception as e:
        print(f"❌ Failed to download IMDB: {e}")
        return None


def download_reuters_dataset():
    """Download Reuters-21578 dataset"""
    try:
        print("📥 Downloading Reuters-21578 dataset...")

        # Use NLTK's Reuters corpus
        import nltk
        nltk.download('reuters', quiet=True)
        from nltk.corpus import reuters

        data = []
        for fileid in reuters.fileids():
            categories = reuters.categories(fileid)
            if categories:  # Skip files without categories
                text = reuters.raw(fileid)
                # Use the first category if multiple
                category = categories[0]
                data.append({'text': text, 'category': category})

        df = pd.DataFrame(data)
        df = df[df['text'].str.len() > 100]  # Remove very short texts

        print(f"✅ Reuters: {len(df)} samples, Size: ~{len(df) * 800 / 1024 / 1024:.1f}MB")
        return df

    except Exception as e:
        print(f"❌ Failed to download Reuters: {e}")
        return None


def download_kaggle_datasets():
    """Download multiple Kaggle datasets"""
    datasets = [
        {
            'id': 'rmisra/news-category-dataset',
            'name': 'News Category',
            'size': '~50MB'
        },
        {
            'id': 'amananandrai/ag-news-classification-dataset',
            'name': 'AG News',
            'size': '~30MB'
        },
        {
            'id': 'clmentbisaillon/fake-and-real-news-dataset',
            'name': 'Fake/Real News',
            'size': '~40MB'
        },
        {
            'id': 'datatattle/covid-19-nlp-text-classification',
            'name': 'COVID-19 Text',
            'size': '~15MB'
        },
        {
            'id': 'kazanova/sentiment140',
            'name': 'Sentiment140 Twitter',
            'size': '~240MB'
        },
        {
            'id': 'snap/amazon-fine-food-reviews',
            'name': 'Amazon Food Reviews',
            'size': '~300MB'
        }
    ]

    downloaded_dfs = []

    if not check_kaggle_setup():
        print("⚠️ Skipping Kaggle datasets - API not configured")
        return downloaded_dfs

    import kaggle

    for dataset in datasets:
        try:
            print(f"\n📦 {dataset['name']} ({dataset['size']})")

            download_path = f"temp_data/{dataset['name'].replace(' ', '_').lower()}"
            os.makedirs(download_path, exist_ok=True)

            # Download dataset
            kaggle.api.dataset_download_files(dataset['id'], path=download_path, unzip=True)

            # Try to load the dataset
            df = load_dataset_from_kaggle_path(download_path, dataset['name'])
            if df is not None and len(df) > 1000:
                downloaded_dfs.append(df)
                print(f"✅ Loaded: {len(df)} samples")
            else:
                print(f"⚠️ Could not load {dataset['name']} or too small")

        except Exception as e:
            print(f"❌ Failed to download {dataset['name']}: {e}")
            continue

    return downloaded_dfs


def load_dataset_from_kaggle_path(path, dataset_name):
    """Load and standardize dataset from Kaggle download"""
    try:
        path = Path(path)
        csv_files = list(path.glob("*.csv"))
        json_files = list(path.glob("*.json"))

        # Try CSV files first
        for csv_file in csv_files:
            try:
                # Read sample first to check size
                sample_df = pd.read_csv(csv_file, nrows=5)
                total_rows = sum(1 for line in open(csv_file, 'r', encoding='utf-8', errors='ignore')) - 1

                # If dataset is very large, sample it
                if total_rows > 500000:
                    skip_rows = sorted(np.random.choice(range(1, total_rows), size=total_rows - 100000, replace=False))
                    df = pd.read_csv(csv_file, skiprows=skip_rows, encoding='utf-8', on_bad_lines='skip')
                else:
                    df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')

                # Standardize columns based on dataset
                df = standardize_dataset_columns(df, dataset_name)

                if df is not None and 'text' in df.columns and 'category' in df.columns:
                    df = df.dropna(subset=['text', 'category'])
                    df = df[df['text'].str.len() > 20]  # Remove very short texts
                    return df[['text', 'category']]

            except Exception as e:
                print(f"   ❌ Error reading {csv_file}: {e}")
                continue

        # Try JSON files
        for json_file in json_files[:1]:
            try:
                df = pd.read_json(json_file, lines=True)
                df = standardize_dataset_columns(df, dataset_name)

                if df is not None and 'text' in df.columns and 'category' in df.columns:
                    df = df.dropna(subset=['text', 'category'])
                    df = df[df['text'].str.len() > 20]
                    return df[['text', 'category']]

            except Exception as e:
                print(f"   ❌ Error reading {json_file}: {e}")
                continue

    except Exception as e:
        print(f"   ❌ Error processing {path}: {e}")

    return None


def standardize_dataset_columns(df, dataset_name):
    """Standardize column names for different datasets"""
    try:
        dataset_name = dataset_name.lower()

        # News Category Dataset
        if 'news' in dataset_name and 'category' in dataset_name:
            if 'headline' in df.columns and 'category' in df.columns:
                df = df.rename(columns={'headline': 'text'})
            elif 'short_description' in df.columns and 'category' in df.columns:
                df = df.rename(columns={'short_description': 'text'})

        # AG News
        elif 'ag' in dataset_name:
            if 'Description' in df.columns and 'Class Index' in df.columns:
                df = df.rename(columns={'Description': 'text', 'Class Index': 'category'})
                # Map numeric categories to names
                category_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
                df['category'] = df['category'].map(category_map)

        # Fake/Real News
        elif 'fake' in dataset_name or 'real' in dataset_name:
            if 'text' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'label': 'category'})
            elif 'title' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'title': 'text', 'label': 'category'})

        # Sentiment140 Twitter
        elif 'sentiment' in dataset_name or 'twitter' in dataset_name:
            if len(df.columns) >= 6:  # Sentiment140 format
                df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
                df['category'] = df['polarity'].map({0: 'negative', 4: 'positive'})
                df = df[['text', 'category']]

        # Amazon Reviews
        elif 'amazon' in dataset_name:
            if 'Text' in df.columns and 'Score' in df.columns:
                df = df.rename(columns={'Text': 'text'})
                # Convert scores to sentiment
                df['category'] = df['Score'].apply(
                    lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')
            elif 'reviewText' in df.columns and 'overall' in df.columns:
                df = df.rename(columns={'reviewText': 'text'})
                df['category'] = df['overall'].apply(
                    lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')

        # COVID-19 Text
        elif 'covid' in dataset_name:
            if 'OriginalTweet' in df.columns and 'Sentiment' in df.columns:
                df = df.rename(columns={'OriginalTweet': 'text', 'Sentiment': 'category'})

        return df

    except Exception as e:
        print(f"   ❌ Error standardizing columns: {e}")
        return df


def download_huggingface_datasets():
    """Download datasets from Hugging Face"""
    try:
        print("📥 Attempting to download from Hugging Face...")

        # Try to install and use datasets library
        try:
            import datasets
        except ImportError:
            print("   Installing datasets library...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            import datasets

        hf_datasets = []

        # IMDB dataset
        try:
            print("   Loading IMDB dataset from Hugging Face...")
            dataset = datasets.load_dataset("imdb", split="train")
            df = pd.DataFrame(dataset)
            df['category'] = df['label'].map({0: 'negative', 1: 'positive'})
            df = df[['text', 'category']]
            hf_datasets.append(df)
            print(f"   ✅ IMDB: {len(df)} samples")
        except Exception as e:
            print(f"   ❌ IMDB failed: {e}")

        # Yelp Reviews
        try:
            print("   Loading Yelp Reviews dataset...")
            dataset = datasets.load_dataset("yelp_review_full", split="train")
            df = pd.DataFrame(dataset)
            # Convert 1-5 stars to categories
            df['category'] = df['label'].map({
                0: 'very_negative', 1: 'negative', 2: 'neutral',
                3: 'positive', 4: 'very_positive'
            })
            df = df[['text', 'category']]
            hf_datasets.append(df)
            print(f"   ✅ Yelp: {len(df)} samples")
        except Exception as e:
            print(f"   ❌ Yelp failed: {e}")

        # AG News
        try:
            print("   Loading AG News dataset...")
            dataset = datasets.load_dataset("ag_news", split="train")
            df = pd.DataFrame(dataset)
            df['category'] = df['label'].map({
                0: 'World', 1: 'Sports', 2: 'Business', 3: 'Science'
            })
            df = df[['text', 'category']]
            hf_datasets.append(df)
            print(f"   ✅ AG News: {len(df)} samples")
        except Exception as e:
            print(f"   ❌ AG News failed: {e}")

        return hf_datasets

    except Exception as e:
        print(f"❌ Hugging Face datasets failed: {e}")
        return []


def combine_and_save_datasets(dataframes):
    """Combine all real datasets and save"""
    if not dataframes:
        print("❌ No datasets were downloaded successfully!")
        return False

    print(f"\n🔄 Combining {len(dataframes)} real datasets...")

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = shuffle(combined_df, random_state=42)

    # Clean and standardize categories
    combined_df['category'] = combined_df['category'].str.lower().str.strip()

    # Remove very short or very long texts
    initial_count = len(combined_df)
    combined_df = combined_df[
        (combined_df['text'].str.len() >= 50) &
        (combined_df['text'].str.len() <= 10000)
        ]
    print(f"   Filtered out {initial_count - len(combined_df)} texts (too short/long)")

    # Keep only categories with enough samples
    category_counts = combined_df['category'].value_counts()
    valid_categories = category_counts[category_counts >= 500].index
    combined_df = combined_df[combined_df['category'].isin(valid_categories)]

    # Save the large dataset
    output_file = "bbc-text-large.csv"
    combined_df.to_csv(output_file, index=False)

    # Create smaller sample for main script
    sample_size = min(50000, len(combined_df))
    sample_df = combined_df.sample(n=sample_size, random_state=42)
    sample_df.to_csv("bbc-text.csv", index=False)

    # Get file sizes
    large_size = os.path.getsize(output_file) / (1024 * 1024)
    small_size = os.path.getsize("bbc-text.csv") / (1024 * 1024)

    print(f"\n🎉 Successfully created real datasets!")
    print(f"📊 Large dataset: {len(combined_df):,} samples")
    print(f"📂 {output_file}: {large_size:.1f} MB")
    print(f"📂 bbc-text.csv: {small_size:.1f} MB")
    print(f"🏷️ Categories ({len(combined_df['category'].unique())}): {list(combined_df['category'].unique())}")
    print(f"\n📈 Category distribution:")
    print(combined_df['category'].value_counts().head(15).to_string())

    # Clean up temp files
    import shutil
    if os.path.exists("temp_data"):
        shutil.rmtree("temp_data")

    return True


def main():
    """Main function to download real large datasets"""
    print("📈 Real Large Text Classification Dataset Downloader")
    print("🌐 Downloading ONLY real datasets from internet sources")
    print("🎯 Target: 300MB+ of real training data")
    print("=" * 70)

    all_datasets = []

    # 1. Download 20 Newsgroups (built-in)
    newsgroups = download_20newsgroups()
    if newsgroups is not None:
        all_datasets.append(newsgroups)

    # 2. Download IMDB Reviews
    imdb = download_imdb_reviews()
    if imdb is not None:
        all_datasets.append(imdb)

    # 3. Download Reuters
    reuters = download_reuters_dataset()
    if reuters is not None:
        all_datasets.append(reuters)

    # 4. Download from Hugging Face
    hf_datasets = download_huggingface_datasets()
    all_datasets.extend(hf_datasets)

    # 5. Download from Kaggle
    kaggle_datasets = download_kaggle_datasets()
    all_datasets.extend(kaggle_datasets)

    # Combine and save
    if all_datasets:
        success = combine_and_save_datasets(all_datasets)
        if success:
            print("\n✅ Real large dataset creation completed!")
            print("🚀 You can now run: python bbc_text_classification.py")
        else:
            print("❌ Failed to combine datasets")
    else:
        print("❌ No real datasets could be downloaded!")
        print("💡 Try setting up Kaggle API or check internet connection")


if __name__ == "__main__":
    main()