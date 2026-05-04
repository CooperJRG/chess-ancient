package alphazerochess;

import alphazerochess.dualresidual.DualResnetModel;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class AlphaZeroChessTrainer {

    private static final Logger log = LoggerFactory.getLogger(AlphaZeroChessTrainer.class);

    public static void main(String[] args) {

        int miniBatchSize = 32;
        int boardSize = 8;

        int numResidualBlocks = 20;
        int numFeaturePlanes = 14;

        log.info("Initializing AGZ model");
        ComputationGraph model = DualResnetModel.getModel(numResidualBlocks, numFeaturePlanes);

        log.info("Create dummy data");
        INDArray input = Nd4j.create(miniBatchSize, numFeaturePlanes, boardSize, boardSize);

        // the value network outputs a value between 0 and 1 to assess the position
        INDArray valueOutput = Nd4j.create(miniBatchSize, 1);

        // Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        // Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        // Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        int listenerFrequency = 1;
        model.setListeners(new StatsListener(statsStorage, listenerFrequency));

        // Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        log.info("Train AGZ model");
        model.fit(new INDArray[] {input}, new INDArray[] {valueOutput});

        // To access the UI, open your browser and go to http://localhost:9000/train
    }
}
