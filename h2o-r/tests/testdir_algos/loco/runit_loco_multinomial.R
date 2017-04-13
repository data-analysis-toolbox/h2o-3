setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

test.loco.multinomial <- function(){
    iris.hex <- as.h2o(iris)

    #Build GBM (Multinomial)
    gbm <- h2o.gbm(1:4,5,iris.hex)

    #Build DRF (Multinomial)
    drf <- h2o.randomForest(1:4,5,iris.hex)

    #Build GLM (Multinomial)
    glm <- h2o.glm(1:4,5,iris.hex,family = "multinomial")

    #Build DL (Multinomial)
    dl <- h2o.deeplearning(1:4,5,iris.hex)

    #GBM LOCO (Multinomial)
    cat("GBM LOCO (Multinomial)\n")
    h2o_loco_default <- h2o.loco(gbm,iris.hex)
    h2o_loco_mean <- h2o.loco(gbm,iris.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(gbm,iris.hex, replace_val = "median")

    cat("H2O LOCO, Default, GBM\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, GBM\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, GBM\n")
    print(h2o_loco_median)

    #DRF LOCO (Multinomial)
    cat("DRF LOCO (Multinomial)\n")
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
    cat("GLM LOCO (Multinomial)\n")
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
    cat("DL LOCO (Multinomial)\n")
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

doTest("LOCO Test", test.loco.multinomial)