package hex.mli;

import hex.genmodel.utils.DistributionFamily;
import hex.mli.loco.LeaveOneCovarOut;
import hex.tree.gbm.GBM;
import org.junit.BeforeClass;
import org.junit.Test;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;
import hex.tree.gbm.GBMModel;
import static hex.genmodel.utils.DistributionFamily.gaussian;
import static hex.genmodel.utils.DistributionFamily.multinomial;
import static hex.genmodel.utils.DistributionFamily.bernoulli;

/*
This Junit is mainly used to detect leaks in Leave One Covariate Out (LOCO)
 */
public class LeaveOneCovarOutTest extends TestUtil {

    @BeforeClass()
    public static void setup() { stall_till_cloudsize(1); }

    @Test
    public void testLocoRegression() {
        //Regression case
        locoRun("./smalldata/junit/cars.csv", "economy (mpg)", gaussian);
    }

    @Test
    public void testLocoBernoulli() {
        //Bernoulli case
        locoRun("./smalldata/logreg/prostate.csv", "CAPSULE", bernoulli);

    }

    @Test
    public void testLocoMultinomial(){
        locoRun("./smalldata/junit/cars.csv", "cylinders", multinomial);
    }

    public Frame locoRun(String fname, String response, DistributionFamily family) {
        GBMModel gbm = null;
        Frame fr = null;
        Frame fr2= null;
        Frame loco=null;
        try {
            Scope.enter();
            fr = parse_test_file(fname);
            int idx = fr.find(response);
            if (family == DistributionFamily.bernoulli || family == DistributionFamily.multinomial || family == DistributionFamily.modified_huber) {
                if (!fr.vecs()[idx].isCategorical()) {
                    Scope.track(fr.replace(idx, fr.vecs()[idx].toCategoricalVec()));
                }
            }
            DKV.put(fr);             // Update frame after hacking it

            GBMModel.GBMParameters parms = new GBMModel.GBMParameters();
            if( idx < 0 ) idx = ~idx;
            parms._train = fr._key;
            parms._response_column = fr._names[idx];
            parms._ntrees = 5;
            parms._distribution = family;
            parms._max_depth = 4;
            parms._min_rows = 1;
            parms._nbins = 50;
            parms._learn_rate = .2f;
            parms._score_each_iteration = true;

            GBM job = new GBM(parms);
            gbm = job.trainModel().get();

            // Done building model; produce a score column with predictions
            fr2 = gbm.score(fr);
            loco = LeaveOneCovarOut.leaveOneCovarOut(gbm,fr,job._job,null);
            return loco;

        } finally {
            if( fr  != null ) fr.remove();
            if( fr2 != null ) fr2.remove();
            if( gbm != null ) gbm.delete();
            if( loco != null ) loco.remove();
            Scope.exit();
        }
    }

}
