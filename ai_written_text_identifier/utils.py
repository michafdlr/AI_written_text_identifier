
from colorama import Fore, Style
from tensorflow import keras
from ai_written_text_identifier.params import *
import pandas as pd
from pathlib import Path
import numpy as np
from google.cloud import storage
from transformers import AutoTokenizer, TFAutoModel
from joblib import load

def get_path_data() -> Path:
    abs_path = Path(__file__).parents[1].absolute()
    return abs_path / "raw_data"

def load_data(source: str="xl-1542M",
              truncation: bool=True,
              n_rows: int=500_000) -> dict[pd.DataFrame]:
    '''Load the data in dictionary of pandas Dataframes.
    ---
    source: specifies the outputs of a GPT-2 model

    ---
    truncation: specifies if Top-K 40 truncation data is used

    ---
    n_rows: specifies the fraction of train-data loaded. Smaller values for testing the code.'''

    path_data = get_path_data()
    final_data={}
    for split in ["train", "valid", "test"]:
        data={}
        if truncation:
            file_path = path_data / f"{source}-k40.{split}.csv"
        else:
            file_path = path_data / f"{source}.{split}.csv"

        if split == "train":
            data['fake'] = pd.read_csv(file_path, usecols=["text"], nrows=n_rows//2) # nrows to have balanced dataset
        else:
            data['fake'] = pd.read_csv(file_path, usecols=["text"])
        data['fake']["AI"] = 1 # AI written

        file_path = path_data / f"webtext.{split}.csv"

        if split == "train":
            data['true'] = pd.read_csv(file_path, usecols=["text"], nrows=n_rows//2) # nrows to have balanced dataset
        else:
            data['true'] = pd.read_csv(file_path, usecols=["text"])
        data['true']["AI"] = 0 # not AI written

        final_data[split] = pd.concat([data["true"], data["fake"]])

    return final_data

# create a function to split datasets into subsets of small, medium and large texts
def divide_frame(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    '''Return tuple of DataFrames with small, medium and large texts
    split according to 1. and 3. quartile of text length distribution.'''
    q = np.percentile(df.text_length, [25, 75])
    q_1, q_3 = q[0], q[1]
    df_small = df[df.text_length < q_1].reset_index(drop=True)
    df_medium = df[df.text_length.between(q_1, q_3)].reset_index(drop=True)
    df_large = df[df.text_length > q_3].reset_index(drop=True)
    return df_small, df_medium, df_large

def model_checkpoint(name: str="distilbert",
                     large: bool=True,
                     uncased: bool=True) -> str:
    model_ckpt = f'{name}-large' if large else f'{name}-base'
    return f'{model_ckpt}-uncased' if uncased else model_ckpt

def load_from_gcs(type, name):
    print(Fore.BLUE + f"\nLoad {name} {type} from GCS..." + Style.RESET_ALL)

    client = storage.Client.from_service_account_json('credentials.json')
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix=f"{type}s/{name}"))
    print(blobs)

    for blob in blobs:
        print(blob.name)
        if type in ('tokenizer', 'extractooor'):
            if not os.path.exists(os.path.join(os.getcwd(),f"{type}s", name)):
                os.mkdir(os.path.join(os.getcwd(),f"{type}s", name))
            path_to_save = os.path.join(os.getcwd(), blob.name)
            print(path_to_save)
        elif type == 'model':
            path_to_save = os.path.join(os.getcwd(),f"{type}s", blob.name.split('/')[-1])
        if type != "extractor":
            blob.download_to_filename(path_to_save)
    if type == 'tokenizer':
        downloaded_object = AutoTokenizer.from_pretrained(os.path.join(os.getcwd(),f"{type}s", name))
    if type == 'extractor':
        downloaded_object = TFAutoModel.from_pretrained(os.path.join(os.getcwd(),f"{type}s", name))
    if type == 'model':
        downloaded_object = load(os.path.join(os.getcwd(),f"{type}s", blob.name.split('/')[-1]))
    print(f"✅ Latest {type} downloaded from cloud storage")
    return downloaded_object

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(os.getcwd(), latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)
        latest_model = keras.models.load_model(latest_model_path_to_save)
        print("✅ Latest model downloaded from cloud storage")

        return latest_model
    except:
        print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        return None
