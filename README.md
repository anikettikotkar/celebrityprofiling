### PAN19 Celebrity Profiling Code
Baseline code from https://github.com/pan-webis-de/pan-code/tree/master/clef20/celebrity-profiling

#### install requirements for python3
* `pip3 install -r requirements.txt`

#### Prepare training data
* Download and unzip PAN2019 dataset from "https://drive.google.com/open?id=1hrMrLDRenBxSKoiLwPC8W68W38Dh9BrE"

#### Prepare training data
Specify `dataset_path` in pre_process.py and terminal run
* `python3 pre_process.py`   
This has already been run and also the data is featurized and pre-saved to ` ./data/loaded_data.npz` 
to avoid computation overhead.   
The data is split into train and test using `TRAIN_COUNT` and `TEST_COUNT` variables.  
(If you change these you need to re-run this and set `pre_load=False` in  `pan20-celebrity-profiling-ngram-baseline.py`)

#### Run baseline training algorithm 
* `python3 celebrity_profiling_ngram_svm_baseline.py`  
Results will be produced in `./results/all-predictons.ndjson`

#### HTML Visualization
An html file called celebrity-profiling-results.html reads in `./results/gt.json` and `./results/pred.json` 
(which contain first 10 examples from the test dataset and their predictions).  
Open the html file to see the results.

#### GUI Visualization
* Run `python3 gui.py` to start the interactive GUI.  
* Enter a tweet in the dialog box and click the button to see the age,gender and occupation of the celebrity predicted by our model.  




