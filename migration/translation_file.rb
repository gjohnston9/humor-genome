table "JokesCleaned" do
  column "joke_id", :key
  column "joke_text", :string, :rename_to => "content"
  column "joke_thumbsupcount", :integer, :rename_to => "upvotes"
  column "joke_thumbsdowncount", :integer, :rename_to => "downvotes"
  column "joke_title", :string, :rename_to => "title"
  column "joke_categories", :string, :rename_to => "categories"
end
