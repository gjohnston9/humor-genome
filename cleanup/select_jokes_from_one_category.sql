SELECT joke_text
FROM JokesCleaned as j
INNER JOIN Jokes2CategoryCleaned as j2c
ON j.joke_id=j2c.joke_id
WHERE j2c.jokecategory_id=140;
