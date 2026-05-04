package com.example.chessv2;

import Data.BoardTrainingData;
import Data.DataReader;
import Network.ChessModelTrainer;
import Network.NeuralNetwork;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Collections;
import java.util.List;

public class Main {

    private static int startIndex = 0;

    public static void main(String[] args) throws IOException, InterruptedException, SQLException {
        // Change these values according to your database configuration
        String dbUrl = "jdbc:sqlite:/Users/coopergilkey/IdeaProjects/ChessV2/src/database.db";

    }

    private static void trainInBatches(NeuralNetwork net, DataReader dR, int batchSize, String URL, int dataSize) {
        List<BoardTrainingData> data = dR.readData(URL,startIndex, dataSize);
        int numberOfBatches = data.size() / batchSize;
        Collections.shuffle(data);
        for (int i = 0; i < numberOfBatches; i++) {
            int fromIndex = i * batchSize;
            int toIndex = (i + 1) * batchSize;
            List<BoardTrainingData> batch = data.subList(fromIndex, toIndex);
            net.train(batch);
        }

        // Train the remaining data if it doesn't fit perfectly into batches
        if (numberOfBatches * batchSize < data.size()) {
            List<BoardTrainingData> remainingData = data.subList(numberOfBatches * batchSize, data.size());
            net.train(remainingData);
        }
    }

}
