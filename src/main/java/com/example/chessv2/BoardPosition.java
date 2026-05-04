package com.example.chessv2;

import java.io.Serial;
import java.io.Serializable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class BoardPosition implements Serializable {

    @Serial
    private static final long serialVersionUID = -6088132223339393324L;
    private final long[] pieces;
    private final boolean currentPlayer; // true for white, false for black
    private final int enPassantAvailableSquare;
    private final boolean[] castlingRights;

    public BoardPosition(Board board) {
        pieces = board.getBitboards().clone(); // Assuming 6 types of pieces for each color
        currentPlayer = board.whiteIsCurrentPlayer(); // White goes first
        enPassantAvailableSquare = board.getEnPassantActiveSquare();
        castlingRights = board.getCastlingRights().clone();
    }


    // Remove the setter methods, keeping only the getter methods
    public long[] getPieces() {
        return pieces;
    }

    public boolean isCurrentPlayer() {
        return currentPlayer;
    }

    public boolean[] getCastlingRights() {
        return castlingRights;
    }

    public int getEnPassantAvailableSquare() {
        return enPassantAvailableSquare;
    }

    public Piece.Color getCurrentPlayer() {
        return currentPlayer ? Piece.Color.WHITE : Piece.Color.BLACK;
    }

    public int getEnPassantActiveSquare() {
        return enPassantAvailableSquare;
    }

    public long getHash() {
        return (long) hashCode();
    }

    @Override
    public boolean equals(Object object){
        if(object instanceof BoardPosition){
            return Arrays.equals(pieces, ((BoardPosition) object).pieces) && Arrays.equals(castlingRights, ((BoardPosition) object).castlingRights)
                    && currentPlayer == ((BoardPosition) object).currentPlayer && enPassantAvailableSquare == ((BoardPosition) object).enPassantAvailableSquare;
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(pieces), currentPlayer, enPassantAvailableSquare, Arrays.hashCode(castlingRights));
    }


    public List<double[][]> getData() {
        List<double[][]> data = new ArrayList<>();

        for (long pieceBitboard : pieces) {
            double[][] pieceData = new double[8][8];
            for (int row = 0; row < 8; row++) {
                for (int col = 0; col < 8; col++) {
                    int squareIndex = row * 8 + col;
                    if ((pieceBitboard & (1L << squareIndex)) != 0) {
                        pieceData[row][col] = 1.0;
                    }
                }
            }
            data.add(pieceData);
        }

        double[][] castlingRightsData = new double[8][8];
        if (castlingRights[0]) castlingRightsData[0][0] = 1.0; // White kingside
        if (castlingRights[1]) castlingRightsData[0][7] = 1.0; // White queenside
        if (castlingRights[2]) castlingRightsData[7][0] = 1.0; // Black kingside
        if (castlingRights[3]) castlingRightsData[7][7] = 1.0; // Black queenside
        if(currentPlayer) castlingRightsData[4][4] = 1.0; // White Current player
        data.add(castlingRightsData);

        double[][] enPassantData = new double[8][8];
        if (enPassantAvailableSquare != -1) {
            int row = enPassantAvailableSquare / 8;
            int col = enPassantAvailableSquare % 8;
            enPassantData[row][col] = 1.0;
        }
        data.add(enPassantData);

        return data;
    }


    @Override
    public String toString() {
        String[] pieceNames = {
                "White Pawns", "White Knights", "White Bishops", "White Rooks", "White Queens", "White Kings",
                "Black Pawns", "Black Knights", "Black Bishops", "Black Rooks", "Black Queens", "Black Kings"
        };

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pieces.length; i++) {
            sb.append(pieceNames[i]).append(":\n");
            double[][] pieceMatrix = getPieceMatrix(pieces[i]);
            for (int row = 7; row >= 0; row--) {
                for (int col = 0; col < 8; col++) {
                    sb.append(String.format("%4.1f", pieceMatrix[row][col]));
                }
                sb.append("\n");
            }
            sb.append("\n");
        }

        sb.append("Castling Rights:\n");
        double[][] castlingMatrix = getCastlingMatrix();
        for (int row = 7; row >= 0; row--) {
            for (int col = 0; col < 8; col++) {
                sb.append(String.format("%4.1f", castlingMatrix[row][col]));
            }
            sb.append("\n");
        }

        sb.append("\nEn Passant:\n");
        double[][] enPassantMatrix = getEnPassantMatrix();
        for (int row = 7; row >= 0; row--) {
            for (int col = 0; col < 8; col++) {
                sb.append(String.format("%4.1f", enPassantMatrix[row][col]));
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    private double[][] getPieceMatrix(long bitboard) {
        double[][] matrix = new double[8][8];

        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                int index = row * 8 + col;
                if ((bitboard & (1L << index)) != 0) {
                    matrix[row][col] = 1.0;
                }
            }
        }

        return matrix;
    }

    private double[][] getCastlingMatrix() {
        double[][] matrix = new double[8][8];

        if (castlingRights[0]) {
            matrix[7][0] = 1.0;
        }
        if (castlingRights[1]) {
            matrix[7][7] = 1.0;
        }
        if (castlingRights[2]) {
            matrix[0][0] = 1.0;
        }
        if (castlingRights[3]) {
            matrix[0][7] = 1.0;
        }

        return matrix;
    }

    private double[][] getEnPassantMatrix() {
        double[][] matrix = new double[8][8];

        if (enPassantAvailableSquare != -1) {
            int row = enPassantAvailableSquare / 8;
            int col = enPassantAvailableSquare % 8;
            matrix[row][col] = 1.0;
        }

        return matrix;
    }

}
