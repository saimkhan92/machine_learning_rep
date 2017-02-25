# This program extracts to 100 TV shows and top 5 characters in the shows
# Stores them in a file

import re
import imdb
import csv
import sys


with open("imdb.txt") as fh:
    text=fh.read()
    #print(text)
    movies=re.findall('title=.*>(.*[a-zA-Z])<',text)

del movies[0]
print(movies)
print(len(movies))

ia = imdb.IMDb()
count=0
movies_temp=movies

with open("movie_shows.txt","w") as fh:
    for i in movies:
        fh.write(str(i)+"\n")

fh=open("tv_characters","w")

try:
    for movie_name in movies_temp:
        count+=1
        print("iteration=="+str(count))
        search_results = ia.search_movie(movie_name)
        if search_results:
             movieID = search_results[0].movieID
             movie = ia.get_movie(movieID)
             if movie:
                 cast = movie.get('cast')
                 topActors = 5
                 for actor in cast[:topActors]:
                     if len(str(actor.currentRole).split())>0:
                         firstname=(str(actor.currentRole).split())[0]
                         fh.write(str(firstname)+"\n")
except:
    fh.close()
    sys.exit()

fh.close()
