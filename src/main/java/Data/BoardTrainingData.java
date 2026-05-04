package Data;

import com.example.chessv2.BoardPosition;

import java.util.List;

public class BoardTrainingData {
    private BoardPosition boardPosition;
    private double winRate;

    public BoardTrainingData(BoardPosition boardPosition, double winRate) {
        this.boardPosition = boardPosition;
        this.winRate = winRate;
    }

    public List<double[][]> getData() {
        return boardPosition.getData();
    }

    public double getWinRate() {
        return winRate;
    }
}

