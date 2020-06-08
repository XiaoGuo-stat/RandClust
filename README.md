## Randomized Spectral Clustering for Large-Scale Networks
### Introduction

**RandClust** performs spectral clustering for large-scale directed and undirected networks using
randomization techniques including the random projection and the random sampling. Specifically, the
random-projection-based SVD (eigendecomposition) or the random-sampling-based SVD (eigendecomposition) is first computed for the 
adjacency matrix of the directed (undirected) network. The k-means or the spherical k-median is 
then performed on the randomized singular (eigen) vectors. 

### Examples

We use the a real network `scoEpinions` to illustrate. The `scoEpinions` object is a sparse matrix 
representing the adjacency matrix of the directed Epinions social network. The largest connected 
component of the original network is collected. There is 75877 nodes and 508836 edges.

```r
data(scoEpinions)
A <- scoEpinions 
```

The random-projection-based SVD of `A` can be computed via

```r
rsvd.pro(A, rank = 3, p = 10, q = 2, dist = "normal", nthread = 1)
```

The random-sampling-based SVD of `A` can be computed via

```r
rsvd.sam(A, P = 0.7, nu = 3, nv = 2)
```

The corresponding random-projection-based and random-sampling-based spectral co-clustering can be
performed respectively using 

```r
rcoclust(A, method = "rsample", ky = 3, kz = 2, P = 0.7, normalize = FALSE)
rcoclust(A, method = "rproject", ky = 3, kz =2, normalize = TRUE)
```

The package also provides a function for sampling a sparse matrix with given probability:

```r
rsample(A, P = 0.5)
```

Similar to directed networks, undirected networks could be handled using `reig.pro`,  `reig.sam`,
`rclust`, `rsample_sym`. An undirected network `youtubeNetwork` is also included. 















