package Network;

import Data.BoardTrainingData;
import Data.Image;
import Layers.Layer;
import com.example.chessv2.BoardPosition;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static Data.MatrixUtility.add;
import static Data.MatrixUtility.multiply;

public class NeuralNetwork implements Serializable{

    List<Layer> _layers;
    double scaleFactor;

    public NeuralNetwork(List<Layer> _layers, double scaleFactor) {
        this._layers = _layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers(){

        if(_layers.size() <= 1){
            return;
        }

        for(int i = 0; i < _layers.size(); i++){
            if(i == 0){
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            } else if (i == _layers.size()-1){
                _layers.get(i).set_previousLayer(_layers.get(i-1));
            } else {
                _layers.get(i).set_previousLayer(_layers.get(i-1));
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, double correctAnswer){
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];

        expected[0] = correctAnswer;

        return add(networkOutput, multiply(expected, -1));
    }

    private int getMaxIndex(double[] in){

        double max = Integer.MIN_VALUE;
        int index = 0;

        for(int i = 0; i < in.length; i++){
            if(in[i] >= max){
                max = in[i];
                index = i;
            }

        }

        return index;
    }

    public double guess(BoardTrainingData board){
        List<double[][]> inList = new ArrayList<>();
        for(double[][] array : board.getData()){
            inList.add(multiply(array, (1.0/scaleFactor)));
        }

        double[] out = _layers.get(0).getOutput(inList);
        double guess = getMaxIndex(out);

        return guess;
    }

    public float test(List<BoardTrainingData> boardTrainingData) {
        int correct = 0;

        for (BoardTrainingData board : boardTrainingData) {
            double guess = guess(board);
            double winRate = board.getWinRate();

            // Check if guess and winRate are equal up to three decimal places
            if (Math.abs(guess - winRate) < 0.001) {
                correct++;
            }
        }

        return (float) correct / boardTrainingData.size();
    }


    public void train (List<BoardTrainingData> boardTrainingData){

        for(BoardTrainingData board:boardTrainingData){
            List<double[][]> inList = new ArrayList<>();
            for(double[][] array : board.getData()){
                inList.add(multiply(array, (1.0/scaleFactor)));
            }

            double[] out = _layers.get(0).getOutput(inList);
            double[] dldO = getErrors(out, board.getWinRate());

            _layers.get((_layers.size()-1)).backPropagation(dldO);
        }

    }

    public static void save(NeuralNetwork network, String filePath) {
        try {
            File file = new File(filePath);
            FileOutputStream fileOut = new FileOutputStream(file, false); // Set append to false to overwrite the file
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(network);
            out.close();
            fileOut.close();
        } catch (IOException i) {
            i.printStackTrace();
        }
    }


    public static NeuralNetwork load(String filePath) {
        NeuralNetwork network = null;
        try {
            FileInputStream fileIn = new FileInputStream(filePath);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            network = (NeuralNetwork) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException i) {
            i.printStackTrace();
        } catch (ClassNotFoundException c) {
            System.out.println("NeuralNetwork class not found");
            c.printStackTrace();
        }
        return network;
    }

}