SELECT joke_id, COUNT(*) as count
FROM Jokes2CategoryCleaned
GROUP BY joke_id
ORDER BY count DESC;