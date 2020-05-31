\name{youtubeNetwork}
\alias{youtubeNetwork}
\title{Youtube social network}
\description{This is a Youtube social network where users form friendship each other.
}
\details{There is 1134890 nodes and 2987624 edges.
}
\usage{
data(youtubeNetwork)
}
\format{
  The \code{youtubeNetwork} object is a sparse matrix representing the adjacency matrix
  of the Youtube social network.
}
\source{\url{http://snap.stanford.edu/data/com-Youtube.html}}
\references{
J. Yang and J. Leskovec. (2012)
\emph{Defining and Evaluating Network Communities based on Ground-truth},
\emph{ICDM}\cr
}
\examples{
data(youtubeNetwork)
A <- youtubeNetwork
reig.sam(A, P=0.7, k = 4)
reig.pro(A, rank = 4)
}


