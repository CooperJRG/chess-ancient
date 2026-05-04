package Network;

import Data.BoardTrainingData;
import com.example.chessv2.BoardPositionDatabase;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ChessModelTrainer {
    private DeepChessModel deepChessModel;

    public ChessModelTrainer() {
        deepChessModel = new DeepChessModel();
    }

    public void trainModelOnBatches(String dbUrl, int epochs, int batchSize) throws SQLException, IOException {
        System.out.println("Loading in data...");
        BoardPositionDatabase boardPositionDatabase = new BoardPositionDatabase(dbUrl);
        List<Long> hashes = boardPositionDatabase.getAllHashes();
        Collections.shuffle(hashes);
        int totalSize = hashes.size();
        int valSize = totalSize / 100;
        int startIndex = valSize;

        ComputationGraph model = deepChessModel.createModel(21, 2, 0.001);
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                .workers(4) // Set the number of workers (GPUs or machines) to use
                .prefetchBuffer(8)
                .build();

        // Initialize training and validation iterators
        DataSetIterator trainIterator = new BoardPositionDataSetIterator(boardPositionDatabase, hashes.subList(startIndex, totalSize), batchSize);
        DataSetIterator valIterator = new BoardPositionDataSetIterator(boardPositionDatabase, hashes.subList(0, startIndex), batchSize);

        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs);

            // Train the model on the training set
            model.fit(trainIterator);

            // Evaluate the model on the validation set
            Evaluation eval = model.evaluate(valIterator);
            System.out.println(eval.stats());

            trainIterator.reset();
            valIterator.reset();
        }

        // Save the best model
        ModelSerializer.writeModel(model, new File("chess_evaluation_best_model.zip"), true);
    }

    public class BoardPositionDataSetIterator implements DataSetIterator {
        private BoardPositionDatabase boardPositionDatabase;
        private List<Long> hashes;
        private int batchSize;
        private int index = 0;
        private int totalSize;

        public BoardPositionDataSetIterator(BoardPositionDatabase boardPositionDatabase, List<Long> hashes, int batchSize) {
            this.boardPositionDatabase = boardPositionDatabase;
            this.hashes = hashes;
            this.batchSize = batchSize;
            this.totalSize = hashes.size();
        }

        @Override
        public DataSet next(int num) {
            List<BoardTrainingData> trainingData = null;
            try {
                trainingData = readData(hashes, index, index + num);
            } catch (SQLException e) {
                throw new RuntimeException(e);
            }

            // Convert trainingData to DataSet
            DataSet dataSet = convertTrainingDataToDataSet(trainingData);

            index += num;

            return dataSet;
        }

        @Override
        public int inputColumns() {
            return 8 * 8 * 14;
        }

        @Override
        public int totalOutcomes() {
            return 1;
        }

        @Override
        public boolean resetSupported() {
            return true;
        }

        @Override
        public boolean asyncSupported() {
            return true;
        }

        @Override
        public void reset() {
            index = 0;
            Collections.shuffle(hashes);
        }

        @Override
        public int batch() {
            return batchSize;
        }

        @Override
        public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
            throw new UnsupportedOperationException("Not implemented");
        }

        @Override
        public org.nd4j.linalg.dataset.api.DataSetPreProcessor getPreProcessor() {
            return null;
        }

        @Override
        public List<String> getLabels() {
            return Collections.singletonList("winRate");
        }

        @Override
        public boolean hasNext() {
            return index < totalSize;
        }

        @Override
        public DataSet next() {
            return next(batchSize);
        }

        public List<BoardTrainingData> readData(List<Long> hashes, int startIndex, int endIndex) throws SQLException {
            List<BoardTrainingData> dataList = new ArrayList<>();

            // Check if endIndex is greater than the total number of data points
            if (endIndex > totalSize) {
                endIndex = totalSize;
            }

            for (int i = startIndex; i < endIndex; i++) {
                dataList.add(new BoardTrainingData(boardPositionDatabase.getPositionByHash(hashes.get(i)),
                        boardPositionDatabase.getLabelByHash(hashes.get(i))));
            }

            return dataList;
        }

        public DataSet convertTrainingDataToDataSet(List<BoardTrainingData> trainingData) {
            int numExamples = trainingData.size();
            int inputSize = 8 * 8 * 14;
            int outputSize = 1;

            // Initialize input and output arrays
            float[][] inputArray = new float[numExamples][inputSize];
            float[][] outputArray = new float[numExamples][outputSize];

            // Fill input and output arrays with training data
            for (int i = 0; i < numExamples; i++) {
                BoardTrainingData data = trainingData.get(i);
                List<double[][]> positionDataSets = data.getData();
                double label = data.getWinRate();
                outputArray[i][0] = (float) label;

                int inputIndex = 0;
                for (double[][] positionData : positionDataSets) {
                    for (int row = 0; row < 8; row++) {
                        for (int col = 0; col < 8; col++) {
                            inputArray[i][inputIndex++] = (float) positionData[row][col];
                        }
                    }
                }
            }

            // Create and return DataSet
            return new DataSet(Nd4j.create(inputArray), Nd4j.create(outputArray));
        }
    }
}

