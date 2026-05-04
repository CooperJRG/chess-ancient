package com.example.chessv2;

import java.io.Serializable;
import java.util.ArrayList;

public class BoardState implements Serializable {
    private long[] pieces;
    private boolean currentPlayer; // true for white, false for black
    private int enPassantAvailableSquare;
    private int halfMoveClock;
    private int fullMoveClock;
    private int repetitionClock;

    private boolean[] castlingRights;

    private PieceContainer whitePieces;
    private PieceContainer blackPieces;

    private long[] pieceMask;

    private ArrayList<Move> legalMoves;

    public BoardState(Board board, boolean temp) {
        pieces = board.getBitboards().clone(); // Assuming 6 types of pieces for each color
        currentPlayer = board.whiteIsCurrentPlayer(); // White goes first
        enPassantAvailableSquare = board.getEnPassantActiveSquare();
        halfMoveClock = board.getHalfMoveClock();
        fullMoveClock = board.getFullMoveClock();
        repetitionClock = board.getRepMoveClock();
        castlingRights = board.getCastlingRights().clone();
        whitePieces = board.getPiecesOfColor(Piece.Color.WHITE).clone();
        blackPieces = board.getPiecesOfColor(Piece.Color.BLACK).clone();
        pieceMask = board.getPieceMasks().clone();
        if(!temp) {
            legalMoves = new ArrayList<>(board.getLegalMoves());
        }
    }

    // Add getter and setter methods for the new attributes
    public int getHalfMoveClock() {
        return halfMoveClock;
    }

    public void setHalfMoveClock(int halfMoveClock) {
        this.halfMoveClock = halfMoveClock;
    }

    public int getFullMoveClock() {
        return fullMoveClock;
    }

    public void setFullMoveClock(int fullMoveClock) {
        this.fullMoveClock = fullMoveClock;
    }

    public int getRepetitionClock() {
        return repetitionClock;
    }

    public void setRepetitionClock(int repetitionClock) {
        this.repetitionClock = repetitionClock;
    }

    public PieceContainer getWhitePieces() {
        return whitePieces;
    }

    public void setWhitePieces(PieceContainer whitePieces) {
        this.whitePieces = whitePieces;
    }

    public PieceContainer getBlackPieces() {
        return blackPieces;
    }

    public void setBlackPieces(PieceContainer blackPieces) {
        this.blackPieces = blackPieces;
    }

    public long[] getPieces() {
        return pieces;
    }

    public void setPieces(long[] pieces) {
        this.pieces = pieces;
    }

    public boolean isCurrentPlayer() {
        return currentPlayer;
    }

    public void setCurrentPlayer(boolean currentPlayer) {
        this.currentPlayer = currentPlayer;
    }

    public boolean[] getCastlingRights() {
        return castlingRights;
    }

    public void setCastlingRights(boolean[] castlingRights) {
        this.castlingRights = castlingRights;
    }

    public int getEnPassantAvailableSquare() {
        return enPassantAvailableSquare;
    }

    public void setEnPassantAvailableSquare(int enPassantAvailableSquare) {
        this.enPassantAvailableSquare = enPassantAvailableSquare;
    }

    public Piece.Color getCurrentPlayer() {
        return currentPlayer ? Piece.Color.WHITE : Piece.Color.BLACK;
    }

    public int getEnPassantActiveSquare() {
        return enPassantAvailableSquare;
    }

    public long[] getPieceMasks() {
        return pieceMask;
    }

    public ArrayList<Move> getLegalMoves() {
        return legalMoves;
    }
}

