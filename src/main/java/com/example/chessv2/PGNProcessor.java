package com.example.chessv2;


import java.io.*;
import java.sql.SQLException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPOutputStream;

public class PGNProcessor implements Serializable{
    @Serial
    private static final long serialVersionUID = 1L;
    public static class ResultInfo implements Serializable{
        @Serial
        private static final long serialVersionUID = 1L;
        int count;
        double sum;

        public ResultInfo() {
            count = 0;
            sum = 0;
        }

        public void addResult(double result) {
            count++;
            sum += result;
        }

        public double getAverage() {
            return sum / count;
        }
    }

    public static List<String> readFile() {
        List<String> results = new ArrayList<>();
        int count = 1;
        try {
            File chessBase = new File("ChessbaseTest.pgn");
            BufferedReader myReader = new BufferedReader(new FileReader(chessBase));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = myReader.readLine()) != null) {
                count++;
                if (!line.startsWith("[")) {
                    sb.append(line).append("\n");
                } else if (sb.length() > 0) {
                    String pgnContent = sb.toString();
                    if(!sb.toString().contains("*")) {
                        results.add(pgnContent);
                    }
                    sb = new StringBuilder();
                }
            }
            if (sb.length() > 0) {
                results.add(sb.toString());
            }
            myReader.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        System.out.println("Lines Read: " + count + ", Results: " + results.size());
        return results;
    }

    public static List<String> readFile(String filePath) {
        List<String> results = new ArrayList<>();
        int count = 1;
        try {
            File chessBase = new File(filePath);
            BufferedReader myReader = new BufferedReader(new FileReader(chessBase));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = myReader.readLine()) != null) {
                count++;
                if (!line.startsWith("[")) {
                    sb.append(line).append("\n");
                } else if (sb.length() > 0) {
                    String pgnContent = sb.toString();
                    if(!sb.toString().contains("*")) {
                        results.add(pgnContent);
                    }
                    sb = new StringBuilder();
                }
            }
            if (sb.length() > 0) {
                results.add(sb.toString());
            }
            myReader.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        System.out.println("Lines Read: " + count + ", Results: " + results.size());
        return results;
    }

    public static void processPgnFile(String filePath, String startingFen, String databaseUrl) {
        AtomicInteger processedCount = new AtomicInteger(0);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");

        ScheduledExecutorService scheduledExecutor = Executors.newSingleThreadScheduledExecutor();
        scheduledExecutor.scheduleAtFixedRate(() -> {
            System.out.println("Games examined: " + processedCount.get() + " at " + LocalDateTime.now().format(formatter));
        }, 1, 1, TimeUnit.MINUTES);

        try {
            List<String> pgnContents = readFile(filePath);
            BoardPositionDatabase boardPositionDatabase = new BoardPositionDatabase(databaseUrl);
            int batchSize = 1_000_000; // Adjust this number based on performance and available resources
            Map<Long, BoardPositionInfo> batchedPositions = new HashMap<>();

            for (String pgnContent : pgnContents) {
                try {
                    PGNParser.PGNResult pgnResult = PGNParser.parsePgn(pgnContent);
                    List<String> moves = pgnResult.getMoves();
                    String resultString = pgnResult.getResult();

                    double resultValue;
                    if (resultString.equals("1-0")) {
                        resultValue = 1.0;
                    } else if (resultString.equals("0-1")) {
                        resultValue = 0.0;
                    } else {
                        resultValue = 0.5;
                    }

                    Board board = new Board(startingFen);
                    for (String moveString : moves) {
                        Move move = new Move(moveString, board, moves);
                        board.makeMove(move, false);
                        BoardPosition position = board.getPosition();
                        long positionHash = position.getHash();

                        BoardPositionInfo positionInfo = batchedPositions.getOrDefault(positionHash, new BoardPositionInfo(position, 0, 0));
                        positionInfo.count += 1;
                        positionInfo.sum += resultValue;
                        batchedPositions.put(positionHash, positionInfo);

                        if (batchedPositions.size() >= batchSize) {
                            updateDatabase(boardPositionDatabase, batchedPositions);
                            batchedPositions.clear();
                        }
                    }

                    processedCount.incrementAndGet();
                } catch (Exception e) {
                    System.out.println("An error occurred while processing a game: " + e.getMessage());
                }
            }

            // Update the remaining positions in the batch
            if (!batchedPositions.isEmpty()) {
                updateDatabase(boardPositionDatabase, batchedPositions);
            }

            scheduledExecutor.shutdown();
        } catch (SQLException e) {
            System.out.println("An error occurred while processing the PGN file.");
            e.printStackTrace();
        }

    }

    public static byte[] serializePosition(BoardPosition position) {
        try {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);

            objectOutputStream.writeObject(position);
            objectOutputStream.close();

            return byteArrayOutputStream.toByteArray();
        } catch (IOException e) {
            System.out.println("Error serializing BoardPosition object: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private static void updateDatabase(BoardPositionDatabase boardPositionDatabase, Map<Long, BoardPositionInfo> batchedPositions) throws SQLException {
        for (Map.Entry<Long, BoardPositionInfo> entry : batchedPositions.entrySet()) {
            long positionHash = entry.getKey();
            BoardPositionInfo positionInfo = entry.getValue();
            byte[] serializedPosition = serializePosition(positionInfo.position);
            double avgResult = positionInfo.sum / positionInfo.count;
            boardPositionDatabase.updatePosition(positionHash, serializedPosition, avgResult);
        }
    }

    private static class BoardPositionInfo {
        BoardPosition position;
        int count;
        double sum;

        public BoardPositionInfo(BoardPosition position, int count, double sum) {
            this.position = position;
            this.count = count;
            this.sum = sum;
        }

    }

    public static void processPGNs(List<String> pgns, String startingFEN, String resultsFile, int batchSize) {
        ConcurrentHashMap<BoardPosition, ResultInfo> results = new ConcurrentHashMap<>();
        AtomicInteger count = new AtomicInteger();


        for (String pgn : pgns) {
                PGNToBoard.PGNBoards positions = PGNToBoard.pgnToFEN(startingFEN, pgn);
                String result = positions.getResult();
                double resultValue = switch (result) {
                    case "1-0" -> 1;
                    case "0-1" -> -1;
                    default -> 0;
                };

                for (BoardPosition position : positions.getBoards()) {
                    results.putIfAbsent(position, new ResultInfo());
                    results.get(position).addResult(resultValue);
                }

                count.getAndIncrement();

                // Save the results every batchSize games
                if (count.get() % batchSize == 0) {
                    synchronized (results) {
                        System.out.println("Saving batch...");
                        saveToFile(results, resultsFile, true);
                        results.clear(); // Clear the results map for the next batch
                    }
                }
        }

        // Shutdown the executor and wait for all tasks to complet

        // Save any remaining results
        if (!results.isEmpty()) {
            saveToFile(results, resultsFile, true);
        }
        for (BoardPosition position : results.keySet()){
            System.out.println(results.get(position).getAverage());
        }
    }

    public static void saveToFile(ConcurrentHashMap<BoardPosition, ResultInfo> results, String fileName, boolean append) {
        try {
            OutputStream fileStream = new FileOutputStream(fileName, append);
            OutputStream gzipStream = new GZIPOutputStream(fileStream);
            ObjectOutputStream objectStream = new ObjectOutputStream(gzipStream);

            objectStream.writeObject(results);

            objectStream.close();
            gzipStream.close();
            fileStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

