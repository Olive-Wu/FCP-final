FCP_assignment_v3.py 

Introduction:
This Python project consists of several tasks related to network analysis and agent-based models. Tasks include implementing and testing the Ising model, the Defuant model, and creating and analyzing random, circular, and small-world networks.


Usage Installations:
1.Ising Model: 
Run the ‘Ising model’ simulation with the ‘-ising_model’ flag.
‘-alpha’: Specifies the alpha parameter 
‘-external’: specifies the strength of the external magnetic field.
 Example: 
$ python3 FCP_assignment_v3.py -ising_model
$ python3 FCP_assignment_v3.py -ising_model -external -0.1
$ python3 FCP_assignment_v3.py -ising_model -alpha 10
$ python3 FCP_assignment_v3.py -test_ising'

2.Defuant Model: 
Run the Defuant model simulation using the ‘-defuant’ flag.
‘-beta’: specifies the beta parameter (update rate).
‘-threshold’: specifies the interaction threshold.
 Example：
$ python3 FCP_assignment_v3.py -defuant
$ python3 FCP_assignment_v3.py -defuant -beta 0.1
$ python3 FCP_assignment_v3.py -defuant -threshold 0.3
$ python3 FCP_assignment_v3.py -test_defuant

3.Network Creation and Analysis:
Create and analyze the network using the ‘-network’ flag.
Provides the number of nodes and optional connection probabilities.
 Example：
    $ python3 FCP_assignment_v3.py -network 10 
Mean degree: <number>
Average path length: <number>
Clustering co-efficient: <number>
$ python3 FCP_assignment_v3.py -test_network

4.Ring and Small-World Networks:
Run the Defuant model simulation using the -defuant flag.
‘-beta: specifies the beta parameter (update rate).
‘-threshold: specifies the interaction threshold.
 Example：
$ python3 FCP_assignment_v3.py -ring_network 10 
$ python3 FCP_assignment_v3.py -small_world 10 
$ python3 FCP_assignment_v3.py -small_world 10 -re_wire 0.1 


Requirements:
•numpy
•matplotlib
•sys


git@github.com:Olive-Wu/FCP-final.git


Contributors:
Olive Wu, Jiamin Xia, Siheng Wang, Zixiang Wang

