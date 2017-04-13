setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

test.loco.binomial <- function(){
    pros.hex <- h2o.uploadFile(locate("smalldata/prostate/prostate.csv.zip"))

    #Build GBM (Binomial)
    gbm <- h2o.gbm(3:9,2,pros.hex)

    #Build DRF (Binomial)
    drf <- h2o.randomForest(3:9,2,pros.hex)

    #Build GLM (Binomial)
    glm <- h2o.glm(3:9,2,pros.hex,family = "binomial")

    #Build DL (Binomial)
    dl <- h2o.deeplearning(3:9,2,pros.hex)

    #GBM LOCO (Binomial)
    cat("GBM LOCO (Binomial)\n")
    h2o_loco_default <- h2o.loco(gbm,pros.hex)
    h2o_loco_mean <- h2o.loco(gbm,pros.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(gbm,pros.hex, replace_val = "median")

    cat("H2O LOCO, Default, GBM\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, GBM\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, GBM\n")
    print(h2o_loco_median)

    #DRF LOCO (Binomial)
    cat("DRF LOCO (Binomial)\n")
    h2o_loco_default <- h2o.loco(drf,pros.hex)
    h2o_loco_mean <- h2o.loco(drf,pros.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(drf,pros.hex, replace_val = "median")

    cat("H2O LOCO, Default, DRF\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, DRF\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, DRF\n")
    print(h2o_loco_median)

    #GLM LOCO
    cat("GLM LOCO (Binomial)\n")
    h2o_loco_default <- h2o.loco(glm,pros.hex)
    h2o_loco_mean <- h2o.loco(glm,pros.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(glm,pros.hex, replace_val = "median")

    cat("H2O LOCO, Default, GLM\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, GLM\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, GLM\n")
    print(h2o_loco_median)

    #DL LOCO
    cat("DL LOCO (Binomial)\n")
    h2o_loco_default <- h2o.loco(dl,pros.hex)
    h2o_loco_mean <- h2o.loco(dl,pros.hex, replace_val="mean")
    h2o_loco_median <- h2o.loco(dl,pros.hex, replace_val = "median")

    cat("H2O LOCO, Default, DL\n")
    print(h2o_loco_default)
    cat("H2O LOCO, Mean replacement, DL\n")
    print(h2o_loco_mean)
    cat("H2O LOCO, Median replacement, DL\n")
    print(h2o_loco_median)
}

doTest("LOCO Test", test.loco.binomial)