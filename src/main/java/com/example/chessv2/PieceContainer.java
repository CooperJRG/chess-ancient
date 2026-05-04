package com.example.chessv2;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;

public class PieceContainer implements Iterable<Piece> {
    private final HashMap<Integer, Piece> positionToPieceMap;
    private final LinkedList<Piece> piecesList;

    public PieceContainer() {
        positionToPieceMap = new HashMap<>();
        piecesList = new LinkedList<>();
    }

    public void add(Piece piece) {
        int position = piece.getPosition();
        positionToPieceMap.put(position, piece);
        piecesList.add(piece);
    }

    public PieceContainer clone() {
        PieceContainer clonedContainer = new PieceContainer();
        for (Piece piece : piecesList) {
            Piece clonedPiece = new Piece(piece.getType(), piece.getColor(), piece.getPosition());
            clonedContainer.add(clonedPiece);
        }
        return clonedContainer;
    }


    public Piece getPieceAtPosition(int position) {
        return positionToPieceMap.get(position);
    }

    public void updatePiecePosition(int oldPosition, int newPosition) {
        Piece piece = positionToPieceMap.get(oldPosition);
        if (piece != null) {
            piece.setPosition(newPosition);
            positionToPieceMap.remove(oldPosition);
            positionToPieceMap.put(newPosition, piece);
        }
    }

    public void updatePieceType(int position, Piece.Type type) {
        Piece piece = positionToPieceMap.get(position);
        if (piece != null) {
            piece.setType(type);
        }
    }

    public void removePiece(int position) {
        Piece piece = positionToPieceMap.get(position);
        if (piece != null) {
            piecesList.remove(piece);
            positionToPieceMap.remove(position);
        }
    }

    @Override
    public Iterator<Piece> iterator() {
        return piecesList.iterator();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("PieceContainer {");
        for (Piece piece : piecesList) {
            sb.append(piece.toString()).append(piece.getType()).append(piece.getPosition());
            sb.append(", ");
        }
        if (!piecesList.isEmpty()) {
            sb.delete(sb.length() - 2, sb.length()); // Remove the trailing comma and space
        }
        sb.append("}");
        return sb.toString();
    }
}

