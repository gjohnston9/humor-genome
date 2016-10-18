UPDATE Jokes2CategoryCleaned
SET jokecategory_name = 
(SELECT c.jokecategory_name
FROM Jokes2CategoryCleaned as j2c
INNER JOIN CategoriesCleaned as c
ON  j2c.jokecategory_id=c.jokecategory_id);
