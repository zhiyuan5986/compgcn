# CompGCN
Note that part of these codes are from [OPENHGNN](https://github.com/BUPT-GAMMA/OpenHGNN)
## Usage

**First**, add dataset folder `cs3319-02-project-1-graph-based-recommendation` just under the root directory.

**Second**, run
```
python /utils/preprocess.py
```
then you will find `demo_graph.bin` (as training set) and `test_graph.bin` (as test set) under `./graph`

**Third**, check the result. In root directory,
```
python main.py --use_best_config
```
then you can see log file and `.pt` file under `./checkpoints`, `score.npy` and `CompGCN_results.csv` under `./output`.
