# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:25:16 2020

@author: Hajar
"""
import re
import nltk
nltk.download('punkt')
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import LancasterStemmer as stemmer


from nltk import word_tokenize
def cleanWord(chaine):
    #Enlever les ponctuations
    chaine = re.sub(u"[^\w\d\s]+", " ", chaine)
    
    #Découper les mots (Tokenization)
    tokens = word_tokenize(chaine)
    
    #Mettre les mots en miniscule
    for i in tokens:
        i=i.lower()
        #Eliminer les top Words
        if i in STOPWORDS:
            continue
        
        #Stemming des mots
        stemming = stemmer()
        i = stemming.stem(i)
        
        #Retourner la liste des mots résultats sous forme d'une chaine
        liste=[]
        liste.append(i)
        return (" ".join(liste))
        

#Ouvrir le fichier mode lecture(r)
fichier1 = open("tweets.txt", "r")

#Lire la 1ère ligne du fichier entré
L1 = fichier1.readline()

#Ovrir le fichier mode ecriture
fichier2= open("new_tweets.txt", "w")

#Ecrire la 1ère ligne dans le fichier résultat
fichier2.write(L1)

#Lire le reste des lignes




        
    
ch="Hellhow are you i'm fine, thanks."
cleanWord(ch)