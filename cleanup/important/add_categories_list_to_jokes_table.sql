UPDATE JokesCleaned
SET joke_categories =
(SELECT GROUP_CONCAT(jokecategory_name)
FROM Jokes2CategoryCleaned as j2c
WHERE j2c.joke_id=JokesCleaned.joke_id);