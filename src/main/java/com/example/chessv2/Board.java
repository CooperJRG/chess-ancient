package com.example.chessv2;

import java.lang.reflect.Array;
import java.util.*;

import static com.example.chessv2.Piece.indexToType;

public class Board {
    // Instance variables

    public static final Piece.Type[] pieceCheckList = new Piece.Type[]{Piece.Type.ROOK, Piece.Type.BISHOP, Piece.Type.NIGHT, Piece.Type.PAWN};
    private final int whiteKingStart = 4;
    private final int blackKingStart = 60;
    private final int whiteKingRookStart = 7;
    private final int whiteQueenRookStart = 0;
    private final int blackKingRookStart = 63;
    private final int blackQueenRookStart = 56;


    private Stack<BoardState> boardStateStack;

    private MoveGenerator moveGenerator = new MoveGenerator();


    private long[] chessPieces;

    private ArrayList<Move> legalMoves;

    private PieceContainer blackPieces;
    private PieceContainer whitePieces;
    private long[] pieceMasks;
    private int halfMoveClock;
    private int fullMoveClock;
    private int repetitionClock;
    private Piece.Color currentPlayer;
    private boolean[] castlingRights;
    private int enPassantActiveSquare;
    private Zobrist zobrist;

    // Indexes for chessPieces
    final int WP = 0; // White Pawns
    final int WN = 1; // White Knights
    final int WB = 2; // White Bishops
    final int WR = 3; // White Rooks
    final int WQ = 4; // White Queens
    final int WK = 5; // White Kings

    final int BP = 6; // Black Pawns
    final int BN = 7; // Black Knights
    final int BB = 8; // Black Bishops
    final int BR = 9; // Black Rooks
    final int BQ = 10; // Black Queens
    final int BK = 11; // Black Kings

    // Indexes for castlingRights
    private int whiteKingRights = 0;
    private int whiteQueenRights = 1;
    private int blackKingRights = 2;
    private int blackQueenRights = 3;

    // Indexes for Piece Masks
    final int WHITE_PIECES = 0;

    final int BLACK_PIECES = 1;

    public Board(String fen) {
        boardStateStack = new Stack<>();
        zobrist = new Zobrist();
        chessPieces = new long[12];
        pieceMasks = new long[2];
        castlingRights = new boolean[]{true, true, true, true};
        fenToBitboard(fen);
        setPieceMasks();
        setPieceContainers();
        setLegalMoves();
    }

    private void setPieceContainers() {
        whitePieces = new PieceContainer();
        blackPieces = new PieceContainer();
        for (int i = 0; i < chessPieces.length; i++) {
            for (int j = 0; j < 64; j++){
                if(((chessPieces[i] >> j) & 1) == 1){
                    if(i < 6){
                        whitePieces.add(new Piece(i, j));
                    } else{
                        blackPieces.add(new Piece(i, j));
                    }
                }
            }
        }
    }

    private void setLegalMovesTest() {
        ArrayList<Move> result = new ArrayList<>();
        EnumMap<Piece.Type, Long> totalTime = new EnumMap<>(Piece.Type.class);
        EnumMap<Piece.Type, Integer> totalCount = new EnumMap<>(Piece.Type.class);

        for (Piece.Type type : Piece.Type.values()) {
            totalTime.put(type, 0L);
            totalCount.put(type, 0);
        }

        for (int i = 0; i < 1000; i++) {
            PieceContainer pieces = whiteIsCurrentPlayer() ? whitePieces : blackPieces;

            for (Piece piece : pieces) {
                long startTime = System.nanoTime();
                List<Move> moves = moveGenerator.generateMoves(piece, this);
                long elapsedTime = System.nanoTime() - startTime;

                totalTime.put(piece.getType(), totalTime.get(piece.getType()) + elapsedTime);
                totalCount.put(piece.getType(), totalCount.get(piece.getType()) + 1);
                result.addAll(moves);
            }
            filterMoves(result);
        }

        legalMoves = result;

        // Print average time for each piece type
        for (Piece.Type type : Piece.Type.values()) {
            if (totalCount.get(type) > 0) {
                double avgTime = totalTime.get(type) / (double) totalCount.get(type);
                System.out.printf("Average time for %s: %.2f ns%n", type, avgTime);
            }
        }

        // Calculate and print average total time
        long sumTotalTime = 0;
        for (Piece.Type type : Piece.Type.values()) {
            sumTotalTime += totalTime.get(type);
        }
        double avgTotalTime = sumTotalTime / 1000.0;
        System.out.printf("Average total time: %.2f ns%n", avgTotalTime);
    }

    private void setLegalMoves() {
        ArrayList<Move> result = new ArrayList<>();
        PieceContainer pieces = whiteIsCurrentPlayer() ? whitePieces : blackPieces;
        for (Piece piece : pieces) {
            List<Move> moves = moveGenerator.generateMoves(piece, this);
            result.addAll(moves);
        }
        filterMoves(result);
        orderMoves(result);
        legalMoves = result;
    }


    private void filterMoves(ArrayList<Move> result) {
        Piece.Color currentPlayer = getCurrentPlayer();
        Piece.Color opponent = currentPlayer.opponent();
        Iterator<Move> moveIterator = result.iterator();
        int kingIndex;
        while (moveIterator.hasNext()) {
            Move move = moveIterator.next();
            // Skip castling move if not allowed
            if (move.getIsCastling() != Move.CastlingType.NONE && !canCastle(currentPlayer, move.getIsCastling())) {
                moveIterator.remove();
                continue;
            }
            // Make the move
            makeMove(move, true);

            // Check if the move puts the current player in check
            kingIndex = kingIndex(currentPlayer);
            if (isInCheck(kingIndex, currentPlayer)) {
                // Remove the move
                moveIterator.remove();
            } else {
                // Cache opponent's check state
                boolean opponentInCheck = isInCheck(kingIndex(opponent), opponent);
                move.setCheck(opponentInCheck);

                if (opponentInCheck) {
                    move.setCheckmate(isInCheckmate(opponent));
                }
            }

            // Restore the board state
            unmakeMove();
        }
    }


        private void orderMoves(ArrayList<Move> result){
        result.sort((move1, move2) -> {
            if (move1.isCheckmate() && !move2.isCheckmate()) {
                return -1;
            } else if (!move1.isCheckmate() && move2.isCheckmate()) {
                return 1;
            } else if (move1.isCheck() && !move2.isCheck()) {
                return -1;
            } else if (!move1.isCheck() && move2.isCheck()) {
                return 1;
            } else if (move1.isCapture() && !move2.isCapture()) {
                return -1;
            } else if (!move1.isCapture() && move2.isCapture()) {
                return 1;
            } else if (move1.isCapture() && move2.isCapture()) {
                int materialDifference1 = move1.getCapturedPiece().getType().getPieceValue() - move1.getMovingPiece().getType().getPieceValue();
                int materialDifference2 = move2.getCapturedPiece().getType().getPieceValue() - move2.getMovingPiece().getType().getPieceValue();

                if (materialDifference1 > materialDifference2) {
                    return -1;
                } else if (materialDifference1 < materialDifference2) {
                    return 1;
                }
            } else if (move1.getIsCastling() != Move.CastlingType.NONE && move2.getIsCastling() == Move.CastlingType.NONE) {
                return -1;
            } else if (move1.getIsCastling() == Move.CastlingType.NONE && move2.getIsCastling() != Move.CastlingType.NONE) {
                return 1;
            }

            return 0;
        });
    }



    // Converts a fen string to a binary representation.
    private void fenToBitboard(String fen) {
        String[] fenParts = fen.split(" ");
        String piecePositions = fenParts[0];
        String activeColor = fenParts[1];
        String castlingRights = fenParts[2];
        String enPassant = fenParts[3];

        int index = 56; // Start at the top-left corner of the board
        for (int i = 0; i < piecePositions.length(); i++) {
            char c = piecePositions.charAt(i);
            if (Character.isDigit(c)) {
                index += Character.getNumericValue(c);
            } else if (c == '/') {
                index -= 16;
            } else {
                long binary = 1L << index;
                switch (c) {
                    case 'P' -> chessPieces[WP] |= binary;
                    case 'N' -> chessPieces[WN] |= binary;
                    case 'B' -> chessPieces[WB] |= binary;
                    case 'R' -> chessPieces[WR] |= binary;
                    case 'Q' -> chessPieces[WQ] |= binary;
                    case 'K' -> chessPieces[WK] |= binary;
                    case 'p' -> chessPieces[BP] |= binary;
                    case 'n' -> chessPieces[BN] |= binary;
                    case 'b' -> chessPieces[BB] |= binary;
                    case 'r' -> chessPieces[BR] |= binary;
                    case 'q' -> chessPieces[BQ] |= binary;
                    case 'k' -> chessPieces[BK] |= binary;
                }
                index++;
            }
        }

        currentPlayer = "w".equals(activeColor) ? Piece.Color.WHITE : Piece.Color.BLACK;

        // Update castling rights
        if (!castlingRights.contains("K")) {
            this.castlingRights[whiteKingRights] = false;
        }
        if (!castlingRights.contains("Q")) {
            this.castlingRights[whiteQueenRights] = false;
        }
        if (!castlingRights.contains("k")) {
            this.castlingRights[blackKingRights] = false;
        }
        if (!castlingRights.contains("q")) {
            this.castlingRights[blackQueenRights] = false;
        }

        enPassantActiveSquare = enPassantTargetSquareToInt(enPassant);
    }

    private int enPassantTargetSquareToInt(String enPassant) {
        if (enPassant.equals("-")) {
            return -1;
        }
        int file = enPassant.charAt(0) - 'a';
        int rank = enPassant.charAt(1) - '1';
        return rank * 8 + file;
    }

    public String[][] bitboardsToArray() {
        String[][] arrayBoard = new String[8][8];
        int index = 64;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                index--;
                if (((chessPieces[WP] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  P  ");
                } else if (((chessPieces[WN] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  N  ");
                } else if (((chessPieces[WB] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  B  ");
                } else if (((chessPieces[WR] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  R  ");
                } else if (((chessPieces[WK] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  K  ");
                } else if (((chessPieces[WQ] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  Q  ");
                } else if (((chessPieces[BP] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  p  ");
                } else if (((chessPieces[BN] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  n  ");
                } else if (((chessPieces[BB] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  b  ");
                } else if (((chessPieces[BR] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  r  ");
                } else if (((chessPieces[BK] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  k  ");
                } else if (((chessPieces[BQ] >> index) & 1) == 1) {
                    arrayBoard[i][j] = ("  q  ");
                } else {
                    arrayBoard[i][j] = ("     ");
                }
            }
        }
        String[][] arrayBoardTwo = new String[8][8];
        for (int i = 0; i < arrayBoard.length; i++) {
            for (int j = 0; j < arrayBoard.length; j++) {
                arrayBoardTwo[i][j] = arrayBoard[i][7 - j];
            }
        }
        return arrayBoardTwo;
    }

    @Override
    public String toString() {
        String[][] arrayBoard = bitboardsToArray();
        StringBuilder sb = new StringBuilder();

        // Draw the top border
        sb.append("+-----+-----+-----+-----+-----+-----+-----+-----+\n");

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                // Draw the left border for each cell
                sb.append("|");

                // If there is a piece, draw the piece; otherwise, draw a space
                sb.append(arrayBoard[i][j].isEmpty() ? "    " : arrayBoard[i][j]);
            }
            // Draw the right border and the rank number
            sb.append("| ").append(9 - (i + 1)).append("\n");

            // Draw the horizontal border between ranks
            sb.append("+-----+-----+-----+-----+-----+-----+-----+-----+\n");
        }

        // Draw the file letters
        sb.append("   a     b     c     d     e     f     g     h\n");

        return sb.toString();
    }



    private void setPieceMasks() {
        pieceMasks[WHITE_PIECES] = 0L;
        pieceMasks[BLACK_PIECES] = 0L;

        for (int i = WP; i <= WK; i++) {
            pieceMasks[WHITE_PIECES] |= chessPieces[i];
        }

        for (int i = BP; i <= BK; i++) {
            pieceMasks[BLACK_PIECES] |= chessPieces[i];
        }
    }

    // Empty methods with comments about their purpose
    public void makeMove(Move move, boolean temp) {
        boardStateStack.push(new BoardState(this, temp));
        halfMoveClock++;
        enPassantActiveSquare = -1;
        if (whiteIsCurrentPlayer()) {
            fullMoveClock++;
        }
        // Make the move on the board
        Move.CastlingType castling = move.getIsCastling();
        Piece movingPiece = move.getMovingPiece();
        Piece.Type movingType = movingPiece.getType();
        int startSquare = move.getStartSquare();
        int endSquare = move.getEndSquare();
        PieceContainer currentPieces = whiteIsCurrentPlayer() ? whitePieces : blackPieces;
        PieceContainer oppPieces = whiteIsCurrentPlayer() ? blackPieces : whitePieces;
        if (castling != Move.CastlingType.NONE) {
            // Handle castling
            handleCastling(move, currentPieces);
        } else {
            Piece capturedPiece = move.getCapturedPiece();
            Piece promotedPiece = move.getPromotionPiece();
            // Move the piece
            switchPieceState(startSquare, movingType, currentPlayer);
            currentPieces.updatePiecePosition(startSquare, endSquare);
            if (capturedPiece != null) { // A piece was captured
                int capturedPos = capturedPiece.getPosition(); // Handles en Passant
                // Capture the piece
                switchPieceState(capturedPos, capturedPiece.getType(), currentPlayer.opponent());
                oppPieces.removePiece(capturedPos);
                halfMoveClock = 0;
            } else {
                // Handle en passant available
                if (movingType == Piece.Type.PAWN && Math.abs(endSquare - startSquare) == 16) {
                    enPassantActiveSquare = currentPlayer == Piece.Color.WHITE ? startSquare + 8 : startSquare - 8;
                }
                // Increment the repetition clock
                if (boardStateStack.size() >= 2) {
                    BoardState prevState = boardStateStack.get(boardStateStack.size() - 2);
                    if (prevState.equals(new BoardState(this, temp))) {
                        repetitionClock++;
                    } else {
                        repetitionClock = 0;
                    }
                }
            }
            // Handle pawn promotion
            if (promotedPiece != null) {
                switchPieceState(endSquare, promotedPiece.getType(), currentPlayer);
                currentPieces.updatePieceType(endSquare, promotedPiece.getType());
            } else {
                switchPieceState(endSquare, movingType, currentPlayer);
            }
        }
        // Update castling rights
        updateCastlingRights(startSquare, currentPlayer);
        changeCurrentPlayer();
        setPieceMasks();
        if(!temp){
            setLegalMoves();
        }
    }

    private void handleCastling(Move move, PieceContainer currentPieces) {
        int startSquare = move.getStartSquare();
        int endSquare = move.getEndSquare();
        int rookStart, rookEnd;
        // Determine rook's start and end positions based on castling type
        if (move.getIsCastling() == Move.CastlingType.KING_SIDE) {
            rookStart = startSquare + 3;
            rookEnd = endSquare - 1;
        } else {
            rookStart = startSquare - 4;
            rookEnd = endSquare + 1;
        }

        // Move the king
        switchPieceState(startSquare, Piece.Type.KING, currentPlayer);
        switchPieceState(endSquare, Piece.Type.KING, currentPlayer);

        // Move the rook
        switchPieceState(rookStart, Piece.Type.ROOK, currentPlayer);
        switchPieceState(rookEnd, Piece.Type.ROOK, currentPlayer);

        // Update the piece container
        currentPieces.updatePiecePosition(startSquare, endSquare);
        currentPieces.updatePiecePosition(rookStart, rookEnd);
    }

    private void updateCastlingRights(int startSquare, Piece.Color color) {
        int index = color == Piece.Color.WHITE ? 0 : 2;
        if(startSquare == blackKingStart || startSquare == whiteKingStart){
            castlingRights[index] = false;
            castlingRights[index + 1] = false;
        } else if(startSquare == whiteKingRookStart || startSquare == blackKingRookStart){
            castlingRights[index] = false;
        } else{
            castlingRights[index + 1] = false;
        }
    }

    private void switchPieceState(int position, Piece.Type pieceType, Piece.Color color) {
        int pieceIndex = getPieceIndex(pieceType, color);
        long positionMask = 1L << position;
        chessPieces[pieceIndex] ^= positionMask;
    }

    private int getPieceIndex(Piece.Type pieceType, Piece.Color color) {
        int result = switch (pieceType) {
            case PAWN -> 0;
            case NIGHT -> 1;
            case BISHOP -> 2;
            case ROOK -> 3;
            case QUEEN -> 4;
            case KING -> 5;
        };

        // Add offset for black pieces
        if (color == Piece.Color.BLACK) {
            result += 6;
        }

        return result;
    }

    private void changeCurrentPlayer() {
        currentPlayer = whiteIsCurrentPlayer() ? Piece.Color.BLACK : Piece.Color.WHITE;
    }

    public void unmakeMove() {
        if (!boardStateStack.isEmpty()) {
            // Pop the last board state from the stack
            BoardState previousState = boardStateStack.pop();

            // Restore the board state
            whitePieces = previousState.getWhitePieces();
            blackPieces = previousState.getBlackPieces();
            setCurrentPlayer(previousState.getCurrentPlayer());
            enPassantActiveSquare = previousState.getEnPassantActiveSquare();
            halfMoveClock = previousState.getHalfMoveClock();
            fullMoveClock = previousState.getFullMoveClock();
            repetitionClock = previousState.getRepetitionClock();
            chessPieces = previousState.getPieces();
            castlingRights = previousState.getCastlingRights();
            pieceMasks = previousState.getPieceMasks();
            legalMoves = previousState.getLegalMoves();
            // Restore other attributes, such as castling rights, if necessary
        } else {
            throw new IllegalStateException("No previous board states available.");
        }
    }

    private void setCurrentPlayer(Piece.Color currentPlayer) {
        this.currentPlayer = currentPlayer;
    }


    public boolean isLegalMove(Move move) {
        // Check if the move is legal
        return legalMoves.contains(move);
    }

    public boolean isInCheck(Piece.Color player) {
        // Check if the given player is in check
        PieceContainer currentPieces = player.isWhite() ? whitePieces : blackPieces;
        int kingIndex = kingIndex(player);
        if(kingIndex == -1){
            return true;
        }
        int kingPos = currentPieces.getPieceAtPosition(kingIndex).getPosition();
        // Check if the king is in check by any type of piece
        if (moveGenerator.generateSlidingMovesCheck(kingPos, Piece.Type.ROOK, this, player)) {
            return true;
        }
        if (moveGenerator.generateSlidingMovesCheck(kingPos, Piece.Type.BISHOP, this, player)) {
            return true;
        }
        if (moveGenerator.generateNightMovesCheck(kingPos, this, player)) {
            return true;
        }
        return moveGenerator.generatePawnMovesCheck(kingPos, this, player);
    }

    public boolean isInCheck(int kingIndex, Piece.Color player) {
        // Check if the given player is in check
        if(kingIndex == -1){
            return true;
        }
        // Check if the king is in check by any type of piece
        if (moveGenerator.generateSlidingMovesCheck(kingIndex, Piece.Type.ROOK, this, player)) {
            return true;
        }
        if (moveGenerator.generateSlidingMovesCheck(kingIndex, Piece.Type.BISHOP, this, player)) {
            return true;
        }
        if (moveGenerator.generateNightMovesCheck(kingIndex, this, player)) {
            return true;
        }
        return moveGenerator.generatePawnMovesCheck(kingIndex, this, player);
    }

    public boolean isInCheckmate(Piece.Color player) {
        List<Move> allMoves = new ArrayList<>();

        // Generate all moves for the player
        for (Piece piece : getPiecesOfColor(player)) {
            allMoves.addAll(moveGenerator.generateMoves(piece, this));
        }

        // Check if any move gets the player out of check
        for (Move move : allMoves) {
            makeMove(move, true);
            if (!isInCheck(player)) {
                unmakeMove();
                return false;
            }
            unmakeMove();
        }

        // If no move gets the player out of check, they are in checkmate
        return true;
    }

    // Checks if a king can castle, assumes the right exists.
    public boolean canCastle(Piece.Color player, Move.CastlingType castlingType) {
        // Get the initial and final positions of the king and the castling direction
        int initialKingPos = kingIndex(player);
        int finalKingPos;
        Piece.Direction castlingDirection;
        int rookPos;
        if (castlingType == Move.CastlingType.KING_SIDE) {
            finalKingPos = initialKingPos + 2;
            castlingDirection = Piece.Direction.EAST;
            rookPos = initialKingPos + 3;
        } else {
            finalKingPos = initialKingPos - 2;
            castlingDirection = Piece.Direction.WEST;
            rookPos = initialKingPos - 4;
        }
        // Check for pieces between the king and the rook
        long occupiedSquaresMask = pieceMasks[WHITE_PIECES] | pieceMasks[BLACK_PIECES];
        for (int pos = Math.min(initialKingPos, rookPos) + 1; pos < Math.max(initialKingPos, rookPos); pos++) {
            if (((1L << pos) & occupiedSquaresMask) != 0) {
                return false;
            }
        }
        // Check if the squares the king moves through are attacked
        for (int pos = initialKingPos; pos != finalKingPos; pos += castlingDirection.getOffset()) {
            if (moveGenerator.generateSlidingMovesCheck(pos, Piece.Type.ROOK, this, player) ||
                    moveGenerator.generateSlidingMovesCheck(pos, Piece.Type.BISHOP, this, player) ||
                    moveGenerator.generateNightMovesCheck(pos, this, player) ||
                    moveGenerator.generatePawnMovesCheck(pos, this, player)) {
                return false;
            }
        }

        return true;
    }



    private int kingIndex(Piece.Color player) {
        if(player != Piece.Color.BLACK){
            for(int i = 0; i < 64; i++) {
                if (((chessPieces[WK] >> i) & 1) == 1){
                    return i;
                }
            }
        } else{
            for(int i = 0; i < 64; i++) {
                if (((chessPieces[BK] >> i) & 1) == 1){
                    return i;
                }
            }
        }
        return -1;
    }

    public boolean isStalemate() {
        // Check if the game is a draw due to stalemate
        return legalMoves.size() == 0 && !isInCheck(currentPlayer);
    }

    public boolean isThreefoldRepetition() {
        // Check if the position has been repeated three times
        return repetitionClock == 6;
    }

    public boolean isFiftyMoveRule() {
        // Check if the fifty-move rule applies
        return halfMoveClock == 50;
    }

    public boolean isInsufficientMaterial() {
        // Check if there is insufficient material for checkmate
        return false;
    }

    // ... Add other necessary getters and setters
    public long getPieceMask(Piece.Color color){
        if(color == Piece.Color.WHITE){
            return pieceMasks[WHITE_PIECES];
        }
        return pieceMasks[BLACK_PIECES];
    }

    public boolean whiteIsCurrentPlayer(){
        return (Piece.Color.WHITE == currentPlayer);
    }
    public Piece.Color getCurrentPlayer(){
        return currentPlayer;
    }


    public Piece getPiece(Piece.Color player, int position) {
        if(player == Piece.Color.WHITE){
            return whitePieces.getPieceAtPosition(position);
        }
        return blackPieces.getPieceAtPosition(position);
    }

    public int getEnPassantActiveSquare() {
        return enPassantActiveSquare;
    }

    public PieceContainer getPiecesOfColor(Piece.Color player) {
        if(player == Piece.Color.WHITE){
            return whitePieces;
        }
        return blackPieces;
    }

    public Move findLegalMove(Move testMove) {
        for(Move move : legalMoves){
            if(move.getStartSquare() == testMove.getStartSquare() && move.getEndSquare() == testMove.getEndSquare()){
                return move;
            }
        }
        return null;
    }

    public boolean canWhiteCastleKingSide() {
        return castlingRights[whiteKingRights];
    }

    public boolean canWhiteCastleQueenSide() {
        return castlingRights[whiteKingRights];
    }

    public boolean canBlackCastleKingSide() {
        return castlingRights[blackKingRights];
    }

    public boolean canBlackCastleQueenSide() {
        return castlingRights[blackQueenRights];
    }

    public ArrayList<Move> getLegalMoves(){
        return legalMoves;
    }

    public long[] getBitboards() {
        return chessPieces;
    }

    public boolean[] getCastlingRights() {
        return castlingRights;
    }

    public int getHalfMoveClock() {
        return halfMoveClock;
    }

    public int getFullMoveClock() {
        return fullMoveClock;
    }

    public int getRepMoveClock() {
        return repetitionClock;
    }

    public long[] getPieceMasks() {
        return pieceMasks.clone();
    }

    public BoardPosition getPosition(){
        return new BoardPosition(this);
    }
}
