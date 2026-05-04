package com.example.chessv2;


import java.util.ArrayList;
import java.util.List;

public class PGNToBoard {

    public static PGNBoards pgnToFEN(String fen, String pgn) {
        List<BoardPosition> chessBoards = new ArrayList<>();

        // Parse the PGN
        PGNParser.PGNResult result = PGNParser.parsePgn(pgn);
        Board tempBoard = new Board(fen);

        // Iterate through the moves
        for (String moveString : result.getMoves()) {
            // Apply the move to the position
            Move tempMove = new Move(moveString, tempBoard, result.getMoves());
            tempBoard.makeMove(tempMove, false);
            chessBoards.add(tempBoard.getPosition());
        }

        return new PGNBoards(chessBoards, result.getResult());
    }
    public static class PGNBoards {
        private List<BoardPosition> boards;
        private String result;

        public PGNBoards(List<BoardPosition> boards, String result) {
            this.boards = boards;
            this.result = result;
        }

        public List<BoardPosition> getBoards() {
            return boards;
        }

        public String getResult() {
            return result;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (BoardPosition chessBoard : boards) {
                sb.append(chessBoard).append(" ");
            }
            sb.append(result);
            return sb.toString();
        }
    }
}


