SET SESSION old_alter_table=1;

ALTER IGNORE TABLE Jokes2CategoryCleaned ADD UNIQUE (joke_id, jokecategory_id);

SET SESSION old_alter_table=0;