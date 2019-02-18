\name{xgbImpute}
\alias{xgbImpute}
\title{test xgbImpute package}
\usage{
xgbImpute(raw.data, learner = 'gblinear', verbose = T)
}
\description{
  Imputation based on XGboost. Fill each column by
  treating it as a regression problem.  For each column i,
  use XGboost to predict i using all other
  columns except i.
}
\arguments{
  \item{raw.data}{a data frame where each row is a different sample}

  \item{learner}{the base learner that the xgboost will use, including 'gblinear' (default) and 'gbtree'}

  \item{verbose}{if TRUE print status updates}
}
\examples{
  x = data.frame(matrix(rnorm(100),10,10))
  x.missing <- x > 1
  x[x.missing] <- NA
  x.imputed <- xgbImpute(x)
}
