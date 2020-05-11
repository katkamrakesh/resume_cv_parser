#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakeshkatkam

Will be training the parser with spacy's named entity recognition(ner)
The data has been gather manually throught multiple resumes from different sources.
New data will be added as data gets available and time permits.
"""

"""
Load the required library
"""
import spacy
import pickle
import random

#-----------------------------------------------------------------------------#

nlp = spacy.blank('en')

#Define the training function
def train_parser(df):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last = True)
        
    for _, annotation in df:
        for ent in annotation['entities']:
            ner.add_label(ent[2])
            
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(10):
            print("Starting iteration "+ str(itn))
            random.shuffle(df)
            losses = {}
            index = 0
            for text, annotations in df:
                try:
                    nlp.update(
                        [text],
                        [annotations],
                        drop = 0.2,
                        sgd = optimizer, #Sigmoid Grad Desc.
                        losses=losses
                    )
                except Exception as e:
                    pass
                
            print(losses)


def main():
    df = pickle.load(open('train_data.pkl','rb'))           
    train_parser(df)
    nlp.to_disk('cv_parser')


if __name__ == '__main__':
    main()

#-----------------------------------------------------------------------------#