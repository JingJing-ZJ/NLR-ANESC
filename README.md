## ANESC-Attributed Network Embedding With Self Cluster

ANESC uses feedforward neural network to model and takes one-hot representation and attributed information of attributed network nodes as input, and after multi-hidden layer learning node's low-dimensional representation, it preserves the node's neighbor topology and potential clustering structure at the output layer.



### Cite
If you use the code, please kindly cite the following paper:
```
@article{ZJ,
  title={Attributed Network Embedding With Self Cluster，ANESC},
  author={ZJ},
  year={2019}
}
```

### Requirements

The codebase is implemented in Python 3.6.5| Anaconda 4.2.0 (64-bit). Package versions used for development are just below.

```
tensorflow        1.4.0
numpy             1.14.3
```

### Datasets

The input to the code consists of two parts: the link file and the property information file.

the link file(directed edge or undirected edge ):

```
undirected edge:
1 2
2 1
1 3
3 1
1 4
4 1
5 6
6 5
...
directed edge:
1 2
1 3
1 4
5 6
...
```

the property information file(The first item  is the node name, and from the second item is the attribute information of the corresponding node in each row):

```
1 142 169 215 257 514 557 679 703 739 760 858 861 999 1002 1024 1081 1130...
5 66 71 96 115 126 141 142 143 149 201 210 228 247 257 267 277 292 294 295 340 341...
125 11 45 60 72 114 115 136 137 142 152 158 167 181 228 229 256 257 277 290...
126 11 62 72 115 126 136 142 154 158 167 213 228 250 251 257 264 277 280 292 294 328...
127 11 45 50 62 115 126 127 142 154 228 250 257 277 292 294 307 328 350 365 367 371 384 386...
...
```

The output of the code is an embedded representation of the node, in the same order as the nodes in the input properties information file:

```
-0.8720349 1.5443019 -0.46409273 -0.9581298 -1.0870069 0.17594178 -1.2041335 1.476207 ...
-0.55781245 -0.47390187 1.2434882 -0.53079826 1.6667682 1.0248817 1.6909162 -0.9179971 ...
0.861871 -0.47989172 0.9026788 -0.8976714 0.58694404 0.6301621 -1.3594122 -1.1731621 ...
-0.06432663 0.9501392 -0.5124597 -0.40079996 -0.1195966 -0.8760405 -1.1351099 -0.3996379 ...
-0.96959436 -0.39352512 0.52864033 -1.3074392 0.8587859 1.1562703 -1.3325124 -1.0983691 ...
...
```

### Options

Learning of the embedding is handled by the `ANESC_runner.py` script which provides the following command line arguments.

```
--data_path          STR       Input graph path.                  Default is `data/washington/`.
--id_dim             INT       Dimension for id_part.             Default is '20'.
--attr_dim           INT       Dimension for attr_part.           Default is '20'.
--alpha            float    Coefficient of attribute mixing.          Default is '0.2'. 
--initial_gamma    float    Coefficient of Initial clustering weight.    Default is '0.001'.
--cluster_number    INT       Number of cluster.                         Default is '5'.
--n_neg_samples'    INT       Number of negative samples.                Default is '10'.
--batch_size        INT       Number of batch_size.                      Default is '64'.
--epoch             INT       Number of epoch.                           Default is '1000'.
```

