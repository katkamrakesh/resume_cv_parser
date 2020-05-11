#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakeshkatkam

This will parse new resume/cv from pre-trained model built in spacy
"""

"""
Load the required library
"""
import spacy
import sys, fitz


class cvParser():
    def __init__(self, fileName=""):
        self.fileName = fileName
    
    def parse(self):
        text = ""
        try:
            #Read the pdf file and convert it into the text#
            doc = fitz.open(self.fileName)
            for page in doc:
                text += str(page.getText())
            text = " ".join(text.split('\n'))
        except Exception as e:
            print(e)
            
        #Load the pre-trained model and parse the given cv#
        nlp_model = spacy.load('cv_parser')
        cv = nlp_model(text)
        for ent in cv.ents:
            print(f'{ent.label_.upper():{30}}-{ent.text}')

def main(fileName):
    obj = cvParser(fileName)
    obj.parse()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else "")