package alphazerochess.dualresidual;

import org.deeplearning4j.nn.graph.ComputationGraph;

public class DualResnetModel {

    public static ComputationGraph getModel(int blocks, int numPlanes) {

        DL4JAlphaZeroChessBuilder builder = new DL4JAlphaZeroChessBuilder();
        String input = "in";

        builder.addInputs(input);
        String initBlock = "init";
        String convOut = builder.addConvBatchNormBlock(initBlock, input, numPlanes, true);
        String towerOut = builder.addResidualTower(blocks, convOut);
        String valueOut = builder.addChessValueHead(towerOut, true);
        builder.addOutputs(valueOut);

        ComputationGraph model = new ComputationGraph(builder.buildAndReturn());
        model.init();

        return model;
    }
}
