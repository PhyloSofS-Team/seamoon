<h3>The directory contains scripts and data for reproducing model training and evaluation.</h3>

<h3>splits</h3>
The files <i>list_train.csv</i>, <i>list_trainx5.csv</i>, <i>list_val.csv</i>, and <i>list_test.csv</i> give the list of collections used for training (base set or augmented set), validating and testing SeaMoon. Each line corresponds to a collection, specified by its identifier (first column) and the query protein used as reference (second column). In the augmented training set, some samples contain the same conformations. For instance, 1A3RL_1SBSL and 1A3RL_7DCXH share the same set of conformations; they differ by the conformation used as reference for inferring the ground-truth motions, either 1SBSL or 7DCXH.   

&#12288;  
The file <i>seq_sim_train_test.m8</i> is the output of evaluating the sequence similarity between train and test proteins using MMseqs2. The file <i>struct_sim_max_train_test.csv</i> gives the best structural hit in the training set, for each test protein. Structural similarity is estimated as TM-score. The file <i>struct_sim_max_train_test_kpax.csv</i> gives the results of the re-evaluation of structural similarity with Kpax.   

The file <i>stats_weighted.csv</i> gives summary statistics computed by DANCE on the union of the augmented training set, validation set and test set. The file <i>stats_test.csv</i> gives summary statistics computed by DANCE on the test set, with three additional columns reporting the similarity to the training set -- <i>id</i> for identity, <i>cov</i> for coverage and <i>tm</i> for TM-score (as computed by TM-align). The file <i>coll_test.csv</i> gives the collectivity values for the ground-truth motions of the test set.   

The file <i>cath_test_all.csv</i> indicates the annotated CATH topologies and superfamilies for the test proteins. The file <i>cath_train_topol_stats.csv</i> reports the occurrences of CAT topologies in the training set. The file <i>list_fold_held_out.txt</i> gives the list of test proteins that do not share any fold (CATH topology) with the training set.   
The file <i>imod_bench_info.txt</i> gives information about the test proteins from the iMod benchmark.  

<h3>results</h3>
The seven CSV files give the NSSE for the best matching pairs of predicted and ground-truth motions, for each test protein and for each method considered, namely the six versions of SeaMoon and the NMA. To identify the best-matching pair, we solve a partial assignment problem, with the constraint that a prediction cannot be paired with more than one ground-truth motion. 

<h3>scripts</h3>
The script <i>filter_and_split.py</i> takes as input a summary statistics file from DANCE and create train, validation, and test sets. The script applies filtering criteria and then randomly splits the resulting subset into 75% for training, 15% for validation and 15% for testing.   

The script <i>anal_motion_pred.R</i> contains R functions used to analyse and plot the results.
