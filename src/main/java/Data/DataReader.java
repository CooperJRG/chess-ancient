package Data;

import com.example.chessv2.BoardPosition;
import com.example.chessv2.BoardPositionDatabase;

import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DataReader {

    private List<Long> hashes;
    private int valSize;
    private int batchSize;


    public List<BoardTrainingData> readData(String dbUrl, int startIndex, int endIndex) {
        List<BoardTrainingData> dataList = new ArrayList<>();
        try {
            BoardPositionDatabase boardPositionDatabase = new BoardPositionDatabase(dbUrl);
            if (hashes == null) {
                hashes = boardPositionDatabase.getAllHashes();
                valSize = hashes.size() / 100;
                //endIndex = valSize;
                batchSize = valSize / 10;
                Collections.shuffle(hashes);
            }

            // Check if endIndex is greater than the total number of data points
            if (endIndex > hashes.size()) {
                endIndex = hashes.size();
            }

            for (int i = startIndex; i < endIndex; i++) {
                dataList.add(new BoardTrainingData(boardPositionDatabase.getPositionByHash(hashes.get(i)),
                        boardPositionDatabase.getLabelByHash(hashes.get(i))));
            }
        } catch (SQLException ex) {
            ex.printStackTrace();
        }

        return dataList;
    }


    public int getSize() {
        return hashes.size();
    }

    public void writeToCSV(String filename, List<BoardTrainingData> trainingData) {
        try {
            // Set the append parameter to false to overwrite the file
            FileWriter writer = new FileWriter(filename, false);
            for (BoardTrainingData data : trainingData) {
                List<double[][]> positionDataSets = data.getData();
                int index = -1;
                double label = data.getWinRate();
                writer.write(label + ",");
                for(double[][] positionData : positionDataSets) {
                    index++;
                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++) {
                            writer.write(positionData[i][j] + "");
                            if (index == positionDataSets.size() - 1 && i == 7 && j == 7) {
                                writer.write("\n");
                            } else {
                                writer.write(",");
                            }
                        }
                    }
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
