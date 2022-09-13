This repository contains an implementation of  a novel Multi-Object Tracker that i've implemented as a part of my MSc Dissertation. The use case is of a fixed frame setting.
The proposed tracker uses a Graph-based approach to model global spatio-temporal patterns. More about the abstract idea can be found in the document [proposal.pdf](https://github.com/ahmedshoaib/mot-gtracker/blob/main/proposal.pdf)
The detailed implementation report can be found here [report.pdf](https://github.com/ahmedshoaib/mot-gtracker/blob/main/report.pdf).


Steps to run:
1. pip install -r test_dev/requirements.txt
2. python main.py

The source code is found in the folder ‘code/’ and ‘test_dev ’ folder contains some
test scripts used during development.
All the .py files in ‘code’ are part of the application that starts using main.py. The
‘models/reid’ contains the ReID [OSNet](https://kaiyangzhou.github.io/deep-person-reid/user_guide) model. And the .png files are outputs of the
application listing the CPU and Mem usage of processes. The ‘typescript’ file is a log
of the command line output of the application during runtime. Here we can see
detailed logs for each video, for both baseline and GTracker method, along with their
throughput and qualitative results.
A code/data folder is expected to be created and have video folders in MOT
Challenge format.
