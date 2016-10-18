SELECT jokecategory_name, jc.jokecategory_id, COUNT(*) as count
FROM JokeCategories as jc
LEFT JOIN Joke2Categories as j2c
ON jc.jokecategory_id=j2c.jokecategory_id
WHERE jc.jokecategory_subcategoryofid=0
GROUP BY jokecategory_name
ORDER BY count DESC;