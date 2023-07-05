import pandas as pd
import numpy as np
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, TFAutoModel
from joblib import load

TOKENIZER = "roberta-large"
EXTRACTOR = "roberta-large"


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def pred(text, tokenizer, extractor, model):
    '''outputs the probability of the text being AI written
    ---
    text_input: text to be classified
    ---
    model_ckpt: model to be used for feature extraction. Options are "distilbert"
    and "roberta".
    ---
    model_name: model to be used for classification. Options are "nn"
    for neural network, "lr" for logistic regression, "ridge" for ridge classifier.
    '''
    # instantiate tokenizer and model
    inputs = tokenizer(text.replace("\n", " "), return_tensors="tf")
    outputs = extractor(**inputs)
    hidden_states = outputs.last_hidden_state[:, 0].numpy()
    # choose model for classification and return prediction and probability
    proba = None
    class_pred = None

    proba = model.predict_proba(hidden_states)[0][1]
    if proba > 0.5:
        class_pred = "AI written"
    else:
        class_pred = "not AI written"
    print(f'Probability of text being AI written: {proba:.2f}. \nThe prediction therfore is that the text is {class_pred}.')
    return proba, class_pred

@app.get("/predict")
def predict(
        text: str,
        model_name: str):
    """
    Make a single prediction depending in a tokenizer, an extracotr and a model among possible values.
    """
    try:
        assert model_name in ['logistic regression',
                          'ridge classifier',
                          'neural network'
                          ]
    except:
        print("You should choose a model among 'logistic regression', 'ridge classifier', 'neural network'")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.getcwd(),"tokenizers", TOKENIZER))
    extractor = TFAutoModel.from_pretrained(os.path.join(os.getcwd(),"extractors", EXTRACTOR))
    model = load(os.path.join(os.getcwd(),"models", f"{model_name.replace(' ', '-')}_{EXTRACTOR}.joblib"))

    proba, class_pred =  pred(text, tokenizer, extractor, model)

    return {
        'proba': proba,
        'class_pred': class_pred
    }


@app.get("/")
def root():
    return {
    'greeting': 'Hello'
}
