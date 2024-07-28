
# GDI
	├─load_config.py
	├─main.py
	├─train_y.py       
	​├─config
	​│  └─ config.yaml    
	​├─models
	​│  ├─ conv.py
	​│  ├─ epred.py
	​│  ├─ esage.py
	​│  ├─ gnn.py
	​│  └─ lstm.py
	​└─utils
	​   ├─ early_stopping.py 
	​   └─ graph_data.py



## Notice

	python.__version__ 3.7
	torch.__version__  1.11.0
	dgl.__version__    0.8.1



# DataSet

We processed the open source IoT traffic data: https://iotanalytics.unsw.edu.au/iottraces.html#.

### 1 Dataset file format description

#### (1) **Device traffic feature**

##### [traffic_simple**.npy] contains simple traffic features, the number of features is 6, encompassing the count of upstream/downstream packets, the mean packet length of the upstream/downstream, and the standard deviation of the upstream/downstream packet length.

- [traffic_simpleA-B.npy] A is the window size, B is the measurement period. For example "traffic_simple5-30" means the window size is 5 and the measurement period is 30 packets.

##### [traffic**.npy] contains complex traffic features, the number of features is 72. These 72 features include 6 packet header length-related features, 7 packet number-related features, 19 packet length-related features, 13 packet arrival interval features, 15 congestion window-related features, and 12 flag bit counting features.

- [trafficA-B.npy] A is the window size and B is the measurement period. For example "traffic5-30" means the window size is 5 and the measurement period is 30 packets.

#### (2) **Device label** 

- [labelA-B.npy] the data label of window size is A and measurement period is B. For example, "label5-30" for "traffic_simple5-30" and "traffic5-30".

### 2 Access to our dataset

**Since the dataset is large and exceeds the free storage quota for git lfs, we upload it elsewhere.**

**The dataset is available at this link：https://pan.baidu.com/s/17cHFbBis9N2P8aYHuRgcZw** 

**Extraction code：qwer**

data/dataset20/** for the first 10 days of data

data/dataset21/** for the last 10 days of data




