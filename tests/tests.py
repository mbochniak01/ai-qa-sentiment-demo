import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

docs = [(list(movie_reviews.words(fileid)), category)
        for fileid in movie_reviews.fileids()
        for category in movie_reviews.categories(fileid)]

print(docs[:3])


