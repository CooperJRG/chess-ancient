package com.example.chessv2;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class PGNParser {

    public static PGNResult parsePgn(String pgn) {
        // Remove comments, variations, and newline characters
        pgn = pgn.replaceAll("\\{[^}]*\\}|;.*|\\(.*?\\)|\n", " ").replaceAll("\\s+", " ").trim();

        Scanner scanner = new Scanner(pgn);
        List<String> moves = new ArrayList<>();
        String result = "";
        String token;

        while (scanner.hasNext()) {
            token = scanner.next();
            if (token.matches("1-0|0-1|1/2-1/2")) {
                result = token;
            } else if (!token.contains(".")) {
                moves.add(token);
            }
        }

        scanner.close();
        return new PGNResult(moves, result);
    }


    public static class PGNResult {
        private List<String> moves;
        private String result;

        public PGNResult(List<String> moves, String result) {
            this.moves = moves;
            this.result = result;
        }

        public List<String> getMoves() {
            return moves;
        }

        public String getResult() {
            return result;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (String move : moves) {
                sb.append(move).append(" ");
            }
            sb.append(result);
            return sb.toString();
        }
    }


}

