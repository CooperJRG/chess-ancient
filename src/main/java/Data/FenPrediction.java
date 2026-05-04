package Data;

import com.example.chessv2.Board;
import com.example.chessv2.BoardPosition;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class FenPrediction {
    private static String condaEnvironmentName = "NeuralNetwork";
    private static String condaBasePath = "/Users/coopergilkey/opt/anaconda3";

    public static void main(String[] args){
        Board test = new Board("r5k1/6pp/4p3/p2p4/P1pqn1Q1/7P/2P3P1/5R1K b - - 0 26");
        BoardPosition data = test.getPosition();
        List<BoardTrainingData> dataList = new ArrayList<>();
        dataList.add(new BoardTrainingData(data, 0));
        DataReader dataReader = new DataReader();
        dataReader.writeToCSV("test_prediction.csv", dataList);
        try {
            // Call the Python script to train the model on the batch
            ProcessBuilder pb = createPythonProcessBuilder("/Users/coopergilkey/PycharmProjects/NeuralNetwork/chessPredictor.py", "test_prediction.csv");
            Process process = pb.start();
            process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static ProcessBuilder createPythonProcessBuilder(String pythonScriptPath, String... args) {
        List<String> command = new ArrayList<>();
        command.add(condaBasePath + "/bin/conda");
        command.add("run");
        command.add("-n");
        command.add(condaEnvironmentName);
        command.add("python");
        command.add(pythonScriptPath);
        Collections.addAll(command, args);

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);

        return pb;
    }
}
