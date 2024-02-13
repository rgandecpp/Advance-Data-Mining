# -------------------------------------------------------------------------
# AUTHOR: Ruchitha Gande
# FILENAME: Similarity.py
# SPECIFICATION: output the two most similar documents according to their cosine similarity
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 40 minutes
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sport, like, one, support, olympic, games]
# --> Add your Python code here
def word_counts(doc):
    a = []
    for term in terms:
        count = 0
        for word in doc.split():
            if word.replace(",","") == term:
                count+=1
        a.append(count)
    return a

document_matrix = []
terms = ["soccer", "favorite", "sport", "like","one", "support", "olympic", "games"]
for each in [doc1,doc2,doc3,doc4]:
    document_matrix.append(word_counts(each))

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here
highest_similarity = 0
for i in range(4):
    for j in range(i+1,4):
        similarity = cosine_similarity([document_matrix[i]], [document_matrix[j]])
        if similarity> highest_similarity:
            highest_similarity = similarity
            most_similar = i,j

# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
# --> Add your Python code here

conversion = ["doc1","doc2","doc3","doc4"]

print("The most similar documents are:{} and {} with cosine similarity = {}".format(conversion[most_similar[0]],conversion[most_similar[1]],highest_similarity[0][0]))