import imdb
ia = imdb.IMDb()
search_results = ia.search_movie('The Walking Dead')


if search_results:
     movieID = search_results[0].movieID
     movie = ia.get_movie(movieID)
     print(movie)
     if movie:
         cast = movie.get('cast')
         topActors = 5
         for actor in cast[:topActors]:
             print (actor.currentRole)
