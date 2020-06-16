import csv, jsonlines
#output_json_filename = output_filename[:output_filename.index(".")] + ".txt"
# These are the column name that are selected to be stored in the csv
keyset = ["story_id", "obs1", "obs2", "hyp1", "hyp2"]
hydrated_tweets = []
# Reads the current tweets
with jsonlines.open('Data/alphanli-train-dev/raw_data/dev.jsonl', "r") as reader:
    for i in reader.iter(type=dict, skip_invalid=True):
        hydrated_tweets.append(i)

# Writes them out
with  open('Data/alphanli-train-dev/test_data.csv', "w+") as output_file:
    d = csv.DictWriter(output_file, keyset)
    d.writeheader()
    d.writerows(hydrated_tweets)