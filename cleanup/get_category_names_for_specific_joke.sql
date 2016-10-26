SELECT j.joke_text, c.jokecategory_name
FROM JokesCleaned as j
INNER JOIN Joke2Categories as j2c on j2c.joke_id=j.joke_id 
INNER JOIN JokeCategories as c on c.jokecategory_id=j2c.jokecategory_id 
WHERE j.joke_id=835;