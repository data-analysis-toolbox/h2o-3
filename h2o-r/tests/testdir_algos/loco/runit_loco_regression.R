setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

test.loco.regression <- function(){
    iris.hex <- as.h2o(iris)
    iris.hex[,5] = as.factor(iris.hex[,5])

    #Build GBM (Regression)
    gbm <- h2o.gbm(c(1:3,5),4,iris.hex)

    #Build DRF (Regression)
    drf <- h2o.randomForest(c(1:3,5),4,iris.hex)

    #Build GLM (Regression)
    glm <- h2o.glm(c(1:3,5),4,iris.hex,family = "gaussian")

    #Build DL (Regression)
    dl <- h2o.deeplearning(c(1:3,5),4,iris.hex)

    #GBM LOCO (Regression)
    cat("GBM LOCO (Regression)\n")
    h2o_loco_default <- h2o.loco(gbm,iris.hex)
    h2o_loco_mean <- h2o.loco(gbm,iris.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(gbm,iris.hex, replace_val = "median")

    cat("H2O LOCO, Default, GBM\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, GBM\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, GBM\n")
    print(h2o_loco_median)

    #DRF LOCO (Regression)
    cat("DRF LOCO (Regression)\n")
    h2o_loco_default <- h2o.loco(drf,iris.hex)
    h2o_loco_mean <- h2o.loco(drf,iris.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(drf,iris.hex, replace_val = "median")

    cat("H2O LOCO, Default, DRF\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, DRF\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, DRF\n")
    print(h2o_loco_median)

    #GLM LOCO
    cat("GLM LOCO (Regression)\n")
    h2o_loco_default <- h2o.loco(glm,iris.hex)
    h2o_loco_mean <- h2o.loco(glm,iris.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(glm,iris.hex, replace_val = "median")

    cat("H2O LOCO, Default, GLM\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, GLM\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, GLM\n")
    print(h2o_loco_median)

    #DL LOCO
    cat("DL LOCO (Regression)\n")
    h2o_loco_default <- h2o.loco(dl,iris.hex)
    h2o_loco_mean <- h2o.loco(dl,iris.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(dl,iris.hex, replace_val = "median")

    cat("H2O LOCO, Default, DL\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, DL\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, DL\n")
    print(h2o_loco_median)
}

doTest("LOCO Test", test.loco.regression)