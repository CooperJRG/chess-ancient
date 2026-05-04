package com.example.chessv2;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class BoardPositionDatabase {
    private Connection connection;


    // Of the format String databaseUrl = "jdbc:sqlite:/Users/coopergilkey/IdeaProjects/ChessV2/src/database.db";
    public BoardPositionDatabase(String databaseUrl) throws SQLException {
        connection = DriverManager.getConnection(databaseUrl);
        createTable();
    }

    private void createTable() throws SQLException {
        String sql = "CREATE TABLE IF NOT EXISTS positions (" +
                "hash BIGINT PRIMARY KEY," +
                "board_position BLOB," +
                "average_result DOUBLE," +
                "count INTEGER" +
                ")";
        try (Statement stmt = connection.createStatement()) {
            stmt.execute(sql);
        }
    }

    public List<Long> getAllHashes() throws SQLException {
        String selectSql = "SELECT hash FROM positions";
        List<Long> hashes = new ArrayList<>();

        try (PreparedStatement selectStmt = connection.prepareStatement(selectSql);
             ResultSet resultSet = selectStmt.executeQuery()) {

            while (resultSet.next()) {
                long hash = resultSet.getLong("hash");
                hashes.add(hash);
            }
        }

        return hashes;
    }

    private BoardPosition deserializePosition(byte[] serializedBoardPosition) {
        try {
            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(serializedBoardPosition);
            ObjectInputStream objectInputStream = new ObjectInputStream(byteArrayInputStream);

            BoardPosition boardPosition = (BoardPosition) objectInputStream.readObject();
            objectInputStream.close();

            return boardPosition;
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Error deserializing BoardPosition object: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    public BoardPosition getPositionByHash(long hash) throws SQLException {
        String selectSql = "SELECT board_position FROM positions WHERE hash = ?";

        try (PreparedStatement selectStmt = connection.prepareStatement(selectSql)) {
            selectStmt.setLong(1, hash);
            try (ResultSet resultSet = selectStmt.executeQuery()) {
                if (resultSet.next()) {
                    byte[] serializedBoardPosition = resultSet.getBytes("board_position");
                    return deserializePosition(serializedBoardPosition);
                }
            }
        }

        return null;
    }

    public void updatePosition(long hash, byte[] serializedBoardPosition, double result) throws SQLException {
        String selectSql = "SELECT average_result, count FROM positions WHERE hash = ?";
        String insertSql = "INSERT INTO positions (hash, board_position, average_result, count) VALUES (?, ?, ?, ?)";
        String updateSql = "UPDATE positions SET board_position = ?, average_result = ?, count = ? WHERE hash = ?";

        try (PreparedStatement selectStmt = connection.prepareStatement(selectSql)) {
            selectStmt.setLong(1, hash);
            try (ResultSet resultSet = selectStmt.executeQuery()) {
                if (resultSet.next()) {
                    double currentAvgResult = resultSet.getDouble("average_result");
                    int count = resultSet.getInt("count");

                    double newAvgResult = ((currentAvgResult * count) + result) / (count + 1);
                    count += 1;

                    try (PreparedStatement updateStmt = connection.prepareStatement(updateSql)) {
                        updateStmt.setBytes(1, serializedBoardPosition);
                        updateStmt.setDouble(2, newAvgResult);
                        updateStmt.setInt(3, count);
                        updateStmt.setLong(4, hash);
                        updateStmt.executeUpdate();
                    }
                } else {
                    try (PreparedStatement insertStmt = connection.prepareStatement(insertSql)) {
                        insertStmt.setLong(1, hash);
                        insertStmt.setBytes(2, serializedBoardPosition);
                        insertStmt.setDouble(3, result);
                        insertStmt.setInt(4, 1);
                        insertStmt.executeUpdate();
                    }
                }
            }
        }
    }

    public double getLabelByHash(Long hash) throws SQLException {
        String selectSql = "SELECT average_result FROM positions WHERE hash = ?";
        try (PreparedStatement selectStmt = connection.prepareStatement(selectSql)) {
            selectStmt.setLong(1, hash);
            try (ResultSet resultSet = selectStmt.executeQuery()) {
                if (resultSet.next()) {
                    Double result = resultSet.getDouble("average_result");
                    return result;
                }
            }
        }
        return -1;
    }
}
