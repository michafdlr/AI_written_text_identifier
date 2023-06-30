import os
import pickle
from pathlib import Path
import transformers
from transformers import (AutoTokenizer,
                          TFAutoModel,
                          TFAutoModelForSequenceClassification
                          )
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from detector.utils import load_data, divide_frame
import tensorflow as tf
from keras import (layers, optimizers, callbacks,
                    Model, losses, metrics, Input, models)
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from joblib import dump, load
from tqdm import tqdm

transformers.utils.logging.set_verbosity_error()


def prepare_datasets():
    train_size = os.environ.get("TRAIN_SIZE")
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    gpt_path = cache_path / "gpt3_output"

    if not (cache_path / f"preprocessed_data_{train_size}").is_dir():
        print("Creating datasets...")
        df_gpt3 = pd.read_csv(gpt_path / "gpt3_simple.csv").reset_index(drop=True)
        df_gpt3_advanced = pd.read_csv(gpt_path / "gpt3_advanced.csv").reset_index(drop=True)
        df_gpt3_advanced["version"] = "gpt3.5"
        df_gpt3_advanced.sample(frac=1,random_state=1).reset_index(drop=True)
        data = load_data()

        train, val, test = (data["train"].reset_index(drop=True),
                        data["valid"].reset_index(drop=True),
                        data["test"].reset_index(drop=True))


        def remove_newline(text: str) -> str:
            return text.replace("\n", " ")

        for df in [train, val, test]:
            df["version"] = np.where(df["AI"] ==1, "gpt2", "human")

        for df in [train, val, test, df_gpt3, df_gpt3_advanced]:
            df["text"] = df["text"].apply(remove_newline)
            df["text_length"] = df['text'].apply(len)

        train = pd.concat([train, df_gpt3.iloc[:140_000, :], df_gpt3_advanced.iloc[:-5000,:]])
        val = pd.concat([val, df_gpt3.iloc[140_000:145_000,:], df_gpt3_advanced.iloc[-5000:-2500,:]])
        test = pd.concat([test, df_gpt3.iloc[145_000:,:], df_gpt3_advanced.iloc[-2500:,:]])

        train_sample = pd.concat(
                        [train[train.version == "human"].sample(n=train_size//2, random_state=1),
                        train[train.version == "gpt2"].sample(n=train_size//8, random_state=1),
                        train[train.version == "gpt3"].sample(n=train_size//8, random_state=1),
                        train[train.version == "gpt3.5"].sample(n=train_size//4, random_state=1)]
                        ).reset_index(drop=True)
        val = pd.concat([val[val.version == "human"].sample(5000, random_state=1),
                        val[val.version == "gpt2"].sample(1250, random_state=1),
                        val[val.version == "gpt3"].sample(1250, random_state=1),
                        val[val.version == "gpt3.5"].sample(2500, random_state=1)]).reset_index(drop=True)
        test = pd.concat([test[test.version == "human"].sample(5000, random_state=1),
                        test[test.version == "gpt2"].sample(1250, random_state=1),
                        test[test.version == "gpt3"].sample(1250, random_state=1),
                        test[test.version == "gpt3.5"].sample(2500, random_state=1)]).reset_index(drop=True)

        ds_train = Dataset.from_pandas(train_sample, split="train")
        ds_val = Dataset.from_pandas(val, split="valid")
        ds_test = Dataset.from_pandas(test, split="test")
        ds_dict = DatasetDict({"train": ds_train, "valid": ds_val, "test": ds_test})
        ds_dict.save_to_disk(cache_path / f'preprocessed_data_{train_size}')
        print(f"Datasets created and saved in {cache_path}!")
    print("Loading datasets from cache")
    ds_dict = load_from_disk(cache_path / f'preprocessed_data_{train_size}')
    print("Datasets loaded from disk!")
    return ds_dict

def create_tokenizer(model_ckpt: str = "roberta-large"):
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    if not (cache_path / f"extractors_tokenizer_{model_ckpt}").is_dir():
        print("Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        tokenizer.save_pretrained(cache_path / f"extractors_tokenizer_{model_ckpt}")
        print(f"Tokenizer saved in {cache_path}!")
    print("Loading tokenizer from cache")
    tokenizer = AutoTokenizer.from_pretrained(cache_path / f"extractors_tokenizer_{model_ckpt}")
    print("Tokenizer loaded from disk!")
    return tokenizer

def encode_data(model_ckpt: str = "roberta-large"):
    train_size = os.environ.get("TRAIN_SIZE")
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    if not (cache_path / f"encoded_data_{train_size}").is_dir():
        print("Encoding data...")
        ds_dict = prepare_datasets()
        tokenizer = create_tokenizer(model_ckpt=model_ckpt)
        ds_encoded = ds_dict.map(tokenize, batched=True, batch_size=10_000)
        ds_encoded.save_to_disk(cache_path / f'encoded_data_{train_size}')
        print(f"Encoded data saved in {cache_path}!")
    print("Loading encoded data from cache")
    ds_encoded = load_from_disk(cache_path / f'encoded_data_{train_size}')
    print("Encoded data loaded from disk!")
    return ds_encoded

def instantiate_extractor(model_ckpt: str = "roberta-large"):
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    if not (cache_path / f"extractors_model_{model_ckpt}").is_dir():
        print("Instantiating extractor...")
        model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
        model.save_pretrained(cache_path / f"extractors_model_{model_ckpt}")
        print(f"Extractor model saved in {cache_path}!")
    print("Loading extractor model from cache")
    model = TFAutoModel.from_pretrained(cache_path / f"extractors_model_{model_ckpt}", from_pt=True)
    print("Extractor model loaded from disk!")
    return model

def get_hidden_states(model_ckpt: str = "roberta-large"):
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    train_size = os.environ.get("TRAIN_SIZE")
    if not (cache_path / f"hidden_states_{model_ckpt}").is_dir():
        print("Extracting hidden states...")
        tokenizer = create_tokenizer(model_ckpt=model_ckpt)
        ds_encoded = encode_data(model_ckpt=model_ckpt)
        model = instantiate_extractor(model_ckpt=model_ckpt)
        def extract_hidden_states(batch):
            inputs = {k: v for k,v in batch.items() if k in tokenizer.model_input_names}
            last_hidden_state = model(**inputs).last_hidden_state
            return {"hidden_state": last_hidden_state[:, 0].numpy()}
        ds_hidden = ds_encoded.map(extract_hidden_states, batched=True, batch_size=50)
        ds_hidden.save_to_disk(cache_path / f'hidden_states_{model_ckpt}_{train_size}')
        print(f"Hidden states saved in {cache_path}!")
    print("Loading hidden states from cache")
    ds_hidden = load_from_disk(cache_path / f'hidden_states_{model_ckpt}_{train_size}')
    print("Hidden states loaded from disk!")
    return ds_hidden

def train_model(model_ckpt: str = "roberta-large",
                model_head: str = "lr",
                lr_C: list=[2**k for k in range(2, 12)],
                ridge_alpha: list=[0.03, 0.035, 0.04, 0.045],
                nn_activation: str = "gelu",
                nn_layers: int = 2,
                nn_neurons: int = 64,
                nn_dropout: float = 0.1,
                nn_init_lr: float = 1e-3,
                nn_batch_size: int = 32):
    '''Train a model on the hidden states extracted from a pretrained model.
    ---
    model_ckpt: specifies the pretrained model to use for extracting hidden states
    ---
    model_head: specifies the model to use for classification. Options are
    'lr': "logistic regression"
    'ridge': "ridge classification
    'nn': "neural network.'''

    # perpare data
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    train_size = os.environ.get("TRAIN_SIZE")
    ds_hidden = get_hidden_states(model_ckpt=model_ckpt)
    X_train = np.array(ds_hidden["train"]["hidden_state"])
    X_val = np.array(ds_hidden["valid"]["hidden_state"])
    X_test = np.array(ds_hidden["test"]["hidden_state"])

    y_train = np.array(ds_hidden["train"]["AI"])
    y_val = np.array(ds_hidden["valid"]["AI"])
    y_test = np.array(ds_hidden["test"]["AI"])
    X_search = np.vstack((X_train, X_val))
    y_search = np.hstack((y_train, y_val))
    split = PredefinedSplit([-1]*X_train.shape[0]+[0]*X_val.shape[0])

    if not any((cache_path / "model_scores").iterdir()):
        scores_dict = {"lr": 0,
                        "ridge": 0,
                        "nn": 0}
        with open(cache_path/"model_scores" / 'scores_dict.pkl', 'wb') as f:
            pickle.dump(scores_dict, f)
    else:
        with open(cache_path/"model_scores" / 'scores_dict.pkl', 'rb') as f:
            scores_dict = pickle.load(f)

    # train model
    if model_head == "lr" or model_head == "ridge":
        print("Training model...")
        if model_head == "lr":
            lr_clf = LogisticRegression(max_iter=5000)
            params = {"C":lr_C}
            search = GridSearchCV(lr_clf,
                                param_grid=params,
                                n_jobs=-1,
                                cv = split,
                                scoring="accuracy")
            print("Search in progress...")
            search.fit(X_search, y_search)
            print(f"Best params: {search.best_params_}")
            best_model = search.best_estimator_
            print("Fitting model...")
            best_model.fit(X_train, y_train)
            score = best_model.score(X_test, y_test)
            if score > scores_dict["lr"]:
                print(f"New test score is {score} which is {score-scores_dict['lr']} better than previous best score!")
                scores_dict["lr"] = score
                with open(cache_path/"model_scores" / 'scores_dict.pkl', 'wb') as f:
                    pickle.dump(scores_dict, f)
                print("Best score saved!")
                dump(best_model, cache_path / f"trained_models_{train_size}" /
                    f"{model_head}_cfl_best_{model_ckpt}.joblib")
                print(f"Best model trained and saved in {cache_path}!")
            else:
                print("New test score is not better than previous best score!")

        if model_head == "ridge":
            ridge_clf = RidgeClassifierCV(alphas=ridge_alpha, cv=split)
            print("Search in progress...")
            ridge_clf.fit(X_search, y_search)
            print(f"Best params: {ridge_clf.best_params_}")
            best_model = ridge_clf
            score = best_model.score(X_test, y_test)
            if score > scores_dict["ridge"]:
                print(f"New test score is {score} which is {score-scores_dict['ridge']} better than previous best score!")
                scores_dict["ridge"] = score
                with open(cache_path/"model_scores" / 'scores_dict.pkl', 'wb') as f:
                    pickle.dump(scores_dict, f)
                print("Best score saved!")
                dump(best_model, cache_path / f"trained_models_{train_size}" /
                    f"{model_head}_clf_best_{model_ckpt}.joblib")
                print(f"Best model trained and saved in {cache_path}!")
            else:
                print("New test score is not better than previous best score!")

    elif model_head == "nn":
        print("Training model...")
        nn_inputs = Input(shape=(X_train.shape[1],))
        for i in range(nn_layers):
            if i == 0:
                x = layers.Dense(nn_neurons, activation=nn_activation,
                                 kernel_initializer="he_normal")(nn_inputs)
                x = layers.Dropout(nn_dropout)(x)
            else:
                x = layers.Dense(nn_neurons, activation=nn_activation,
                                 kernel_initializer="he_normal")(x)
                x = layers.Dropout(nn_dropout)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        nn_model = Model(inputs=nn_inputs, outputs = outputs)

        decay_steps = X_train.shape[0] // nn_batch_size

        lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=nn_init_lr,
                                                            decay_steps=decay_steps,
                                                            decay_rate=0.9)

        es = callbacks.EarlyStopping(monitor="val_binary_accuracy",
                                    mode="max",
                                    patience=10,
                                    restore_best_weights=True)

        nn_model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                        metrics=[metrics.BinaryAccuracy()],
                        optimizer = optimizers.legacy.Adam(lr_schedule))
        print("Model instantiated! Start training...")
        nn_model.fit(X_train, y_train, batch_size=nn_batch_size,
                        epochs=150,
                        validation_data=(X_val, y_val),
                        callbacks=[es])
        score = nn_model.evaluate(X_test, y_test)[1]
        if score > scores_dict["nn"]:
            print(f"New test score is {score} which is {score-scores_dict['nn']} better than previous best score!")
            scores_dict["nn"] = score
            with open(cache_path/"model_scores" / 'scores_dict.pkl', 'wb') as f:
                pickle.dump(scores_dict, f)
            print("Best score saved!")
            best_model = nn_model
            best_model.save(cache_path / f"nn_model_{model_ckpt}_{train_size}")
            print(f"Best model trained and saved in {cache_path}!")
        else:
            print("New test score is not better than previous best score!")
    else:
        raise ValueError("model_head must be 'lr', 'ridge' or 'nn'!")
    return None

def make_prediction():
    pass
