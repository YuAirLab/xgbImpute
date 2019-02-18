# xgbImpute
#
# Imputation using XGBoosted
# Fill each column by treating it as a regression problem.  For each
# column i, use XGboost to predict i using all other
# columns except i.
#
# @param raw.data a data frame where each row is a different sample
# @param learner the base learner that the xgboost will use, including 'gblinear' (default) and 'gbtree'
# @param verbose if True print status updates

xgbImpute <- function(raw.data, learner = 'gblinear', verbose = F) {
  K <- sort(apply(raw.data, MARGIN = 2, FUN = function(x) sum(is.na(x))))
  tune.col <- match(names(K[1]), colnames(raw.data))
  mean.imputed <- meanImpute(x = raw.data)$x
  missings.idx <- meanImpute(x = raw.data)$missing.matrix

  # start from K > 0 column
  K <- K[K > 0]
  K <- match(names(K), colnames(raw.data))

  # imputation
  converge.old <- Inf
  converge.new <- 1
  iter <- 0
  x.new <- mean.imputed
  time.start <- proc.time()
  while ((converge.new < converge.old) & iter < 5) {
    if (iter != 0) {
      converge.old <- converge.new
    }
    if (verbose) cat("xgbImpute iteration ",iter+1,"...")
    x.old <- x.new
    for (s in K) {
      na.rows <- missings.idx[ , s]
      y.obs <- x.new[!na.rows , s]
      x.obs <- x.new[!na.rows, -s]
      time.tune.start <- proc.time()
      dtrain <- xgb.DMatrix(data = as.matrix(x.obs), label = y.obs)
      if (learner == 'gblinear') {
        model <- xgboost(data = dtrain,
                         nthread = 4,
                         booster = "gblinear",
                         nrounds = 100,
                         # alpha = 0.1,
                         lambda = 10,
                         verbose = F
        )
      } else if (learner == 'gbtree') {
        model <- xgboost(data = dtrain,
                         nrounds = 100,
                         nthread = 4,
                         verbose = F,
                         eta = 0.15,
                         gamma = 0,
                         min_child_weight = 1,
                         max_depth = 6,
                         subsample = 0.8,
                         colsample_bytree = 0.8,
                         early_stopping_rounds = 30
        )
      }
      x.mis <- x.new[na.rows, -s]
      y.mis <- predict(model, newdata = as.matrix(x.mis))
      x.new[na.rows, s] <- y.mis
    }
    converge.new <- sum((x.new - x.old)^2)/sum((x.old)^2)
    if (verbose) cat("done.\n")
    iter <- iter + 1
  }
  return(x.new)
}
