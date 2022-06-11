!wget -O moviedataset.zip https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%205/data/moviedataset.zip
print('unziping ...')

!unzip -o -j moviedataset.zip 
#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')
#Head is a function that gets the first N rows of a dataframe. N's default is 5.

#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

#Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')

#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, 
                                     genre] = int(1)
       
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)

#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)

#Filtering out the movies by title
# no iteration done it matches the title in one shot

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.


#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

#clean this up a bit by resetting the index and dropping the movieId, title, genres and year columns.
userMovies=userMovies.drop(['movieId','title','genres','year'],axis=1)
userMovies.reset_index(inplace=True)
userMovies.reset_index(drop=True)
#display(userMovies)
#to start learning the input's preferences!
#To do this, we're going to turn each genre into weights. We can do this by using the input's reviews and multiplying them into the input's genre table and then summing up the resulting table by column.This operation is actually a dot product between a matrix and a vector, so we can simply accomplish by calling the Pandas "dot" function.
userMovies=userMovies.drop(['index'],axis=1)

inputMovies=inputMovies.drop(['movieId','title'],axis=1)
#converting matrix to array
np_array=userMovies.to_numpy()
#transpose
a=np.array([[3.5,2.0,5.0,4.5,5.0]])
a.transpose()
#pre userprofile
arr=np.dot(a,np_array)
#sum of all (to divide)
x=np.sum(arr)
#userprofile
for i in range(0,1):
  for j in range(0,20):
    arr[i][j]/=x
#display(arr)

# extracting the genre table from the original dataframe:
new = arr.reshape(1, 20)
j=moviesWithGenres_df
d=moviesWithGenres_df[['movieId']]
moviesWithGenres_df=moviesWithGenres_df.drop(['movieId','title','genres','year'],axis=1)
#With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted average of every movie based on the input profile and recommend the top twenty movies that most satisfy it.
#transpose
hey=moviesWithGenres_df.transpose()
nte=np.matmul(new,hey)
#transpose
final=nte.transpose()

if [column for column in d.columns] not in [column for column in final.columns]:
    add = final.append(d)
    add = add[[column for column in moviesWithGenres_df.columns if column in add.columns]]

final['movieId']=j['movieId']
final['title']=j['title']
#renaming the columns
final.columns=['rating','movieId','title']
#sorting the rating column in descending order
final.sort_values(by=['rating'],ascending=False,inplace=True)
#display(final)
top20=final.head(20)
display(top20)
