import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_written_text_identifier.utils import load_from_gcs

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

if __name__ == "__main__":
    #app = FastAPI()
    tokenizer = load_from_gcs('tokenizer', 'extractors_tokenizer_roberta-large')
    extractor = load_from_gcs('extractor', 'roberta-large')
    model = load_from_gcs('model', 'lr_clf_best_roberta-large')
    pred("this is a huge hue gkrsg  popjdwdwd d test!", tokenizer, extractor, model)
