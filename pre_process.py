import json
import os

os.makedirs('./data', exist_ok=True)

TRAIN_COUNT = 27000
TEST_COUNT = 6836
dataset_path = "PATH_TO_pan19-celebrity-profiling-training-dataset-2019-01-31"
#dataset_path = "/Users/priyankamannikeri/DP/Final/pan19-celebrity-profiling-training-dataset-2019-01-31"
all_celebs = [json.loads(line) for line in open(dataset_path + "/labels.ndjson", "r")]

def main():
    celeb_dict = {}
    for celeb in all_celebs:
        celeb_dict[celeb["id"]] = celeb

    train_labels = []
    count = 0

    for idx, line in enumerate(open(dataset_path + "/feeds.ndjson", "r", encoding="utf8")):
        if count == TRAIN_COUNT + TEST_COUNT:
            break
        curr_id = int(json.loads(line)["id"])
        train_labels.append(celeb_dict[curr_id])
        count += 1
    # print(count)

    open("./data/gt-labels.ndjson", "w").writelines(
        [json.dumps(x) + "\n" for x in train_labels[0:TRAIN_COUNT+TEST_COUNT]]
    )

main()
