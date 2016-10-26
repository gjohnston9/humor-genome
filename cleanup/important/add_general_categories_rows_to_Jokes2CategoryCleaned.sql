INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 502, "Adult"
FROM Joke2Categories
WHERE jokecategory_id IN (10, 22, 83, 291);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 503, "Classic Structures"
FROM Joke2Categories
WHERE jokecategory_id IN (108, 183, 187, 231, 282, 377, 393, 474);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 504, "Dark Humor/Bad Taste"
FROM Joke2Categories
WHERE jokecategory_id IN (112, 143, 188, 371);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 505, "Stereotypes"
FROM Joke2Categories
WHERE jokecategory_id IN (71, 95, 104, 219, 226, 251, 297);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 506, "Insulting"
FROM Joke2Categories
WHERE jokecategory_id IN (159, 403);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 507, "Observational"
FROM Joke2Categories
WHERE jokecategory_id IN (41, 60, 135, 140);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 508, "Society"
FROM Joke2Categories
WHERE jokecategory_id IN (148, 214, 241, 246, 302, 313, 365, 378, 480);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 509, "Professional"
FROM Joke2Categories
WHERE jokecategory_id IN (1, 122, 193, 202, 208, 260, 283, 303, 394);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 510, "Relationship"
FROM Joke2Categories
WHERE jokecategory_id IN (167, 180, 232, 265);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 511, "Religion/Faith"
FROM Joke2Categories
WHERE jokecategory_id IN (272);

INSERT INTO Jokes2CategoryCleaned
	(joke_id, jokecategory_id, jokecategory_name)
SELECT DISTINCT joke_id, 999, "no category"
FROM Joke2Categories
WHERE jokecategory_id IN (10, 22, 83, 291);