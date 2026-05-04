package com.example.chessv2;

import java.io.IOException;
import java.sql.SQLException;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static com.example.chessv2.PGNProcessor.*;

public class ChessTester {

    private static final long[][] EXPECTED_STATS = {
            {1, 0, 0, 0, 0, 0, 0},
            {6, 0, 0, 0, 0, 0, 0},
            {264, 87, 0, 6, 48, 10, 0},
            {9_467, 1_021, 4, 0, 120, 38, 22},
            {422_333, 131_393, 0, 7795, 60_032, 15_492, 5},
            {15_833_292, 2_046_173, 6_512, 0, 329_464, 200_568, 50_562},
            {706_045_033, 210_369_132, 212, 10882006, 81_102_984, 26_973_664, 81_076}
    };
    private static final String perft5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ";
    private static final String perft4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    private static final String STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    private static final String databaseUrl = "jdbc:sqlite:/Users/coopergilkey/IdeaProjects/ChessV2/src/database.db";

    public static void main(String[] args) {

        String filePath = "Chessbase.pgn";
        String outPutPath = "ChessTraining.dat";
        String startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        int batchSize = 50_000;

        //pgnScan();
        processPgnFile(filePath, startingFen, databaseUrl);
        /*
        try {
            BoardPositionDatabase boardPositionDatabase = new BoardPositionDatabase(databaseUrl);
            List<Long> hashes = boardPositionDatabase.getAllHashes();
            System.out.println(hashes.size());
            for (int i = 0; i < 10 && i < hashes.size(); i++) {
                long hash = hashes.get(i);
                BoardPosition boardPosition = boardPositionDatabase.getPositionByHash(hash);
                System.out.println("Board position #" + (i + 1) + ":");
                System.out.println(boardPosition);
                System.out.println("----------------------");
            }
        } catch (SQLException e) {
            System.out.println("An error occurred while fetching board positions from the database.");
            e.printStackTrace();
        }

         */
    }


    public static void pgnScan(){
        List<String> pgnList = PGNProcessor.readFile();
        String fileName = "/Users/coopergilkey/IdeaProjects/ChessV2/chesspositionsTest.dat";
        Instant startTime = Instant.now();
        AtomicInteger processedGames = new AtomicInteger(0);

        ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor();
        executor.scheduleAtFixedRate(() -> {
            int gamesProcessed = processedGames.get();
            Instant currentTime = Instant.now();
            Duration elapsedTime = Duration.between(startTime, currentTime);
            System.out.println("Processed " + gamesProcessed + " games");
            System.out.printf("Elapsed time: %d minutes, %d seconds%n", elapsedTime.toMinutes(), elapsedTime.toSecondsPart());
        }, 1, 1, TimeUnit.MINUTES);

        for (String pgn : pgnList) {
            PGNProcessor.processPGNs(List.of(pgn), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", fileName, 1000);
            processedGames.incrementAndGet();
        }

        executor.shutdown();

        //Map<BoardPosition, PGNProcessor.ResultInfo> loadedMap = PGNProcessor.loadFromFile(fileName);

        Instant endTime = Instant.now();
        Duration totalTime = Duration.between(startTime, endTime);
        System.out.printf("Total time: %d minutes, %d seconds%n", totalTime.toMinutes(), totalTime.toSecondsPart());
        System.out.println("Completed");
    }

    public static void depthTester(Board board, int maxDepth) {

        System.out.printf("%10s%14s%14s%14s%14s%16s%14s%16s%16s%14s%n",
                "Depth", "Nodes", "Captures", "E.p.", "Castles", "Promotions", "Checks", "Checkmates", "Time Taken", "Result");

        for (int depth = 1; depth <= maxDepth; depth++) {
            Instant start = Instant.now();
            long[] stats = generateMovesAtDepth(board, depth);
            Instant end = Instant.now();

            Duration timeTaken = Duration.between(start, end);
            long hours = timeTaken.toHours();
            long minutes = timeTaken.toMinutes() % 60;
            long seconds = timeTaken.getSeconds() % 60;
            long milliseconds = timeTaken.toMillis() % 1000;

            String timeTakenString = String.format("%02d:%02d:%02d.%03d", hours, minutes, seconds, milliseconds);
            boolean passed = true;
            for (int i = 0; i < stats.length; i++) {
                if (Math.abs(stats[i] - EXPECTED_STATS[depth][i]) > 1) {
                    passed = false;
                    break;
                }
            }
            String status = passed ? "PASSED" : "FAILED";
            System.out.printf("%10d%,14d%,14d%,14d%,14d%,16d%,14d%,16d%16s%14s%n",
                    depth, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], timeTakenString, status);
        }
    }


    private static long[] generateMovesAtDepth(Board board, int depth) {
        if (depth == 0) {
            return new long[]{1, 0, 0, 0, 0, 0, 0};
        }

        long[] stats = new long[7]; // Nodes, Captures, E.p., Castles, Promotions, Checks, Checkmates
        ArrayList<Move> legalMoves = board.getLegalMoves();
        for (Move move : legalMoves) {
            board.makeMove(move, false);
            long[] childStats = generateMovesAtDepth(board, depth - 1);
            board.unmakeMove();

            stats[0] += childStats[0]; // Add nodes from child node
            if (depth == 1) {
                // Only accumulate counts at the current depth
                stats[1] += move.getCapturedPiece() != null ? 1 : 0; // Captures
                stats[2] += move.getWasEnPassant() ? 1 : 0; // E.p.
                stats[3] += move.getIsCastling() != Move.CastlingType.NONE ? 1 : 0; // Castles
                stats[4] += move.getPromotionPiece() != null ? 1 : 0; // Promotions
                stats[5] += move.isCheck() ? 1 : 0; // Checks
                stats[6] += move.isCheckmate() ? 1 : 0; // Checkmates
            } else {
                // Use the counts from the child node
                stats[1] += childStats[1];
                stats[2] += childStats[2];
                stats[3] += childStats[3];
                stats[4] += childStats[4];
                stats[5] += childStats[5];
                stats[6] += childStats[6];
            }
        }
        return stats;
    }

    public static void printMovesByType(ArrayList<Move> moves) {
        HashMap<Piece.Type, ArrayList<Move>> movesByType = new HashMap<>();
        for (Piece.Type type : Piece.Type.values()) {
            movesByType.put(type, new ArrayList<>());
        }

        for (Move move : moves) {
            Piece.Type type = move.getMovingPiece().getType();
            movesByType.get(type).add(move);
        }

        for (Piece.Type type : Piece.Type.values()) {
            System.out.print(type + ": ");
            ArrayList<Move> movesOfType = movesByType.get(type);
            if (movesOfType.isEmpty()) {
                System.out.println("No moves");
            } else {
                StringBuilder movesString = new StringBuilder();
                for (int i = 0; i < movesOfType.size(); i++) {
                    movesString.append(movesOfType.get(i));
                    if (i < movesOfType.size() - 1) {
                        movesString.append(", ");
                    }
                }
                System.out.println(movesString);
            }
        }
    }
}
