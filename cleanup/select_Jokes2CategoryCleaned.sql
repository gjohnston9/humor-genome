SELECT COUNT(*), jokecategory_name FROM Jokes2CategoryCleaned
GROUP BY jokecategory_name
-- WHERE jokecategory_name="Adult";
-- WHERE jokecategory_name="Classic Structures";