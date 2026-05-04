package com.example.chessv2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;

public class Move {

    private final int whiteKingStart = 4;

    private final int blackKingStart = 60;
    private int startSquare;
    private int endSquare;
    private Piece movingPiece;
    private Piece capturedPiece = null; // null if no piece is captured
    private boolean isCheck = false;
    private boolean isCheckmate = false;

    private CastlingType isCastling;

    private Piece promotionPiece = null; // null if no promotion occurs

    public Move(String san, Board board, List<String> moves) {
        stringIsCastle(san, board);
        if(isCastling == CastlingType.NONE) {
            stringFindEndSquare(san);
            if (!stringFindStartSquare(san, board)){
                throw new IllegalArgumentException("Ambiguous move specified in the SAN string: " + san + moves);
            }
        }
    }

    public boolean isCapture() {
        return capturedPiece != null;
    }

    public boolean getWasEnPassant() {
        return capturedPiece != null && capturedPiece.getPosition() != endSquare;
    }


    enum CastlingType{
        KING_SIDE,
        QUEEN_SIDE,
        NONE;
    }

    private Move(Piece piece, int endSquare) {
        // This constructor is only used to test if a move is legal
        this.startSquare = piece.getPosition();
        this.movingPiece = piece;
        this.endSquare = endSquare;
    }

    public Move(Piece movingPiece, int startSquare, int endSquare, Piece capturedPiece, CastlingType isCastling, Piece promotionPiece) {
        this.startSquare = startSquare;
        this.endSquare = endSquare;
        this.movingPiece = movingPiece;
        this.capturedPiece = capturedPiece;
        this.isCastling = isCastling;
        this.promotionPiece = promotionPiece;
    }

    private void stringIsCastle(String san, Board board) {
        // Is a castle move
        if(san.contains("O")){
            // Is a queen side castle
            isCastling = san.contains("O-O-O") ? CastlingType.QUEEN_SIDE : CastlingType.KING_SIDE;
            // If it is a castle move, only need to save the piece and starting position
            // the rest of the behavior will be handled in the board class when the move is made
            if(board.whiteIsCurrentPlayer()){
                movingPiece = board.getPiece(Piece.Color.WHITE, whiteKingStart);
                startSquare = whiteKingStart;
            } else{
                movingPiece = board.getPiece(Piece.Color.BLACK, blackKingStart);
                startSquare = blackKingStart;
            }
            endSquare = startSquare + (isCastling == CastlingType.QUEEN_SIDE ? -2 : 2);
            stringCheckForCheck(san);
        } else{
            isCastling = CastlingType.NONE;
        }
    }

    private Piece.Type charToPieceType(char letter) {
        Piece.Type result = Piece.Type.PAWN;
        switch (letter) {
            case 'R' -> result = Piece.Type.ROOK;
            case 'N' -> result = Piece.Type.NIGHT;
            case 'B' -> result = Piece.Type.BISHOP;
            case 'Q' -> result = Piece.Type.QUEEN;
            case 'K' -> result = Piece.Type.KING;
        }
        return result;
    }

    private void stringCheckForCheck(String san) {
        if(san.contains("+")){
            isCheck = true;
        } else if(san.contains("#")){
            isCheckmate = true;
        }
    }

    private boolean stringFindStartSquare(String san, Board board) {
        Piece.Type movingPieceType = charToPieceType(san.charAt(0));
        Piece.Color currentPlayer = board.getCurrentPlayer();

        PieceContainer possiblePieces = board.getPiecesOfColor(currentPlayer);

        int disambiguationFile = -1;
        int disambiguationRank = -1;
        int lastIndex = san.length() - 1;
        if(Character.isUpperCase(san.charAt(0)) && san.charAt(0) != 'Q'){ // Piece move
            char firstDisambiguation = san.charAt(1);
            char secondDisambiguation = san.charAt(2);
            if(san.length() > 3) {
                if (Character.isDigit(firstDisambiguation)) { // checks if the rank needs disambiguation
                    disambiguationRank = firstDisambiguation - '1';
                } else if (Character.isLowerCase(firstDisambiguation) && firstDisambiguation != 'x' && !Character.isDigit(secondDisambiguation)) {
                    disambiguationFile = firstDisambiguation - 'a';
                }
            }
        } else if (san.charAt(0) == 'Q'){ // Queen move
            char firstDisambiguation = san.charAt(1);
            char secondDisambiguation = san.charAt(2);
            if (Character.isDigit(firstDisambiguation)) { // checks if the rank needs disambiguation
                disambiguationRank = firstDisambiguation - '1';
            } else if (Character.isLowerCase(firstDisambiguation) && firstDisambiguation != 'x' && !Character.isDigit(secondDisambiguation)) {
                disambiguationFile = firstDisambiguation - 'a';
            } else if(rankAndFileToPos(san.substring(1, 3)) != endSquare && Character.isDigit(secondDisambiguation) ){
                disambiguationFile = firstDisambiguation - 'a';
                disambiguationRank = secondDisambiguation - '1';
            }
        } else { // Pawn Move
            char firstDisambiguation = san.charAt(0);
            if(san.charAt(1) == 'x'){
                disambiguationFile = firstDisambiguation - 'a';
            }
        }
        // disambiguationFile and disambiguationRank will be -1 if no disambiguation is needed
        for (Piece piece : possiblePieces) {
            if(piece.getType() == movingPieceType) {
                int piecePos = piece.getPosition();
                // If disambiguation is needed, check if the piece matches that qualifier.
                if (disambiguationFile != -1 && piecePos % 8 != disambiguationFile) {
                    continue;
                }
                if (disambiguationRank != -1 && piecePos / 8 != disambiguationRank) {
                    continue;
                }
                Move testMove = new Move(piece, endSquare);
                Move result = board.findLegalMove(testMove);
                if (result != null) {
                    this.startSquare = result.startSquare;
                    this.movingPiece = result.movingPiece;
                    this.capturedPiece = result.capturedPiece;
                    int promotionIndex = san.indexOf('=') + 1;
                    if(promotionIndex != 0){
                        switch (san.charAt(promotionIndex)) {
                            case 'N' -> this.promotionPiece = new Piece(Piece.Type.NIGHT, board.getCurrentPlayer(), endSquare);
                            case 'B' -> this.promotionPiece = new Piece(Piece.Type.BISHOP, board.getCurrentPlayer(), endSquare);
                            case 'R' -> this.promotionPiece = new Piece(Piece.Type.ROOK, board.getCurrentPlayer(), endSquare);
                            case 'Q' -> this.promotionPiece = new Piece(Piece.Type.QUEEN, board.getCurrentPlayer(), endSquare);
                        }
                    }
                    return true;
                }
            }
        }
        return false;
    }

    private void stringFindEndSquare(String san) {
        // Check if check, checkmate, or promotion, this way we can
        // find the starting index of the substring.
        int start = san.length() - 2;
        start -= san.contains("+") || san.contains("#") ? 1 : 0;
        start -= san.contains("=")  ? 2 : 0;

        // Corrected the substring range
        endSquare = rankAndFileToPos(san.substring(start, start + 2));
    }

    private int rankAndFileToPos(String square){
        int file = square.charAt(0) - 'a';
        int rank = square.charAt(1) - '1';
        return rank * 8 + file;
    }


    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        int startFile = startSquare % 8;
        int startRank = startSquare / 8;
        int endFile = endSquare % 8;
        int endRank = endSquare / 8;

        builder.append((char) ('a' + startFile));
        builder.append((char) ('1' + startRank));
        builder.append((char) ('a' + endFile));
        builder.append((char) ('1' + endRank));

        if (promotionPiece != null) {
            builder.append(Character.toLowerCase(promotionPiece.getType().toString().charAt(0)));
        }

        return builder.toString();
    }

    // Add getters and setters
    public int getStartSquare(){
        return startSquare;
    }

    public int getEndSquare(){
        return endSquare;
    }

    public CastlingType getIsCastling(){
        return isCastling;
    }

    public Piece getMovingPiece(){
        return movingPiece;
    }

    public Piece getCapturedPiece(){
        return capturedPiece;
    }

    public Piece getPromotionPiece(){
        return promotionPiece;
    }

    public boolean isCheck(){
        return isCheck;
    }

    public boolean isCheckmate(){
        return isCheckmate;
    }

    public void setCheck(boolean check) {
        isCheck = check;
    }

    public void setCheckmate(boolean checkmate) {
        isCheckmate = checkmate;
    }
}


