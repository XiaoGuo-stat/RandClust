\name{scoEpinions}
\alias{scoEpinions}
\title{Epinions social network}
\description{This is a who-trust-whom online social network of a general consumer review site. Members of the site can decide whether to ''trust'' each other. All the trust relationships interact and form the Web of Trust which is then combined with review ratings to determine which reviews are shown to the user.
}
\details{The largest connected component of the original network is collected. There is 75877 nodes and 508836 edges.
}
\usage{
data(scoEpinions)
}
\format{
  The \code{scoEpinions} object is a sparse matrix representing the adjacency matrix
  of the Epinions social network.
}
\source{\url{http://snap.stanford.edu/data/soc-Epinions1.html}}
\references{
M. Richardson, R. Agrawal, and P. Domingos. (2003)
\emph{Trust management for the semantic web},
\emph{International semantic Web conference, 351-368}\cr
}
\examples{
data(scoEpinions)
A <- scoEpinions
rsvd.sam(A, P = 0.7, nu = 3, nv = 3)
}


