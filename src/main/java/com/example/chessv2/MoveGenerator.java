package com.example.chessv2;

import java.util.ArrayList;
import java.util.Objects;

public class MoveGenerator {

    // Generates all semi-legal moves.
    public ArrayList<Move> generateMoves(Piece piece, Board board) {
        ArrayList<Move> result = new ArrayList<>();
        Piece.Type type = piece.getType();
        switch (type) {
            case PAWN -> result = generatePawnMoves(piece, board);
            case ROOK, QUEEN, BISHOP -> result = generateSlidingMoves(piece, board);
            case NIGHT -> result = generateNightMoves(piece, board);
            case KING -> result = generateKingMoves(piece, board);
        }
        return result;
    }

    private ArrayList<Move> generateKingMoves(Piece piece, Board board) {
        ArrayList<Move> result = new ArrayList<>();
        // Save the initial position
        int initialPos = piece.getPosition();
        int initialCol = initialPos & 7; // equivalent to % 8, but faster
        int initialRow = initialPos >> 3; // equivalent to / 8, but faster
        Piece.Color player = board.getCurrentPlayer();
        Piece.Color opponent = player.opponent();
        long playerMask = board.getPieceMask(player);
        // Check moves in each direction
        for (Piece.Direction direction : piece.getMoveOptions()) {
            // Update positions
            int newPosition = initialPos + direction.getOffset();
            int newCol = initialCol + direction.getColOffset();
            int newRow = initialRow + direction.getRowOffset();
            // Wraps around, off-board or blocked by same color
            if ((newCol & ~7) != 0 || (newRow & ~7) != 0 || ((1L << newPosition) & playerMask) != 0) {
                continue;
            }
            Piece originalPiece = board.getPiece(opponent, newPosition);
            Piece opponentPiece = originalPiece != null ? originalPiece.clone() : null;
            // Add the move
            result.add(new Move(piece.clone(), initialPos, newPosition, opponentPiece,
                    Move.CastlingType.NONE, null));
        }

        // Handle castling moves
        if (player == Piece.Color.WHITE) {
            if (board.canWhiteCastleKingSide()) {
                result.add(new Move(piece, initialPos, 6, null, Move.CastlingType.KING_SIDE, null));
            }
            if (board.canWhiteCastleQueenSide()) {
                result.add(new Move(piece, initialPos, 2, null, Move.CastlingType.QUEEN_SIDE, null));
            }
        } else {
            if (board.canBlackCastleKingSide()) {
                result.add(new Move(piece, initialPos, 62, null, Move.CastlingType.KING_SIDE, null));
            }
            if (board.canBlackCastleQueenSide()) {
                result.add(new Move(piece, initialPos, 58, null, Move.CastlingType.QUEEN_SIDE, null));
            }
        }

        return result;
    }

    private ArrayList<Move> generateNightMoves(Piece piece, Board board) {
        ArrayList<Move> result = new ArrayList<>();
        // Save the initial position
        int initialPos = piece.getPosition();
        int initialCol = initialPos & 7; // equivalent to % 8, but faster
        int initialRow = initialPos >> 3; // equivalent to / 8, but faster
        Piece.Color player = board.getCurrentPlayer();
        Piece.Color opponent = player.opponent();
        long playerMask = board.getPieceMask(player);

        // Check moves in each direction
        for (Piece.Direction direction : piece.getMoveOptions()) {
            // Updates positions
            int newPosition = initialPos + direction.getOffset();
            int newCol = initialCol + direction.getColOffset();
            int newRow = initialRow + direction.getRowOffset();

            // Wraps around, off-board or blocked by same color.
            if ((newCol & ~7) != 0 || (newRow & ~7) != 0 || ((1L << newPosition) & playerMask) != 0) {
                continue;
            }

            Piece originalPiece = board.getPiece(opponent, newPosition);
            Piece opponentPiece = originalPiece != null ? originalPiece.clone() : null;

            // Add move
            result.add(new Move(piece.clone(), initialPos, newPosition, opponentPiece, Move.CastlingType.NONE, null));
        }

        return result;
    }

    private ArrayList<Move> generatePawnMoves(Piece piece, Board board) {
        ArrayList<Move> result = new ArrayList<>();
        int initialPos = piece.getPosition();
        int initialCol = initialPos & 7; // equivalent to % 8, but faster
        int initialRow = initialPos >> 3; // equivalent to / 8, but faster
        Piece.Color currentPlayer = board.getCurrentPlayer();
        Piece.Color opponent = currentPlayer.opponent();
        long playerMask = board.getPieceMask(currentPlayer);
        long opponentMask = board.getPieceMask(opponent);
        boolean isWhite = currentPlayer.isWhite();

        // Single move forward
        int forwardOffset = isWhite ? 8 : -8;
        int forwardPosition = initialPos + forwardOffset;
        if (((forwardPosition & ~63) == 0) && ((1L << forwardPosition) & (playerMask | opponentMask)) == 0) {
            handlePawnPromotion(piece, initialPos, forwardPosition, -1, board, result);
        }

        // Double move forward
        if ((initialRow == 1 && isWhite) || (initialRow == 6 && !isWhite)) {
            int doubleForwardOffset = isWhite ? 16 : -16;
            int doubleForwardPosition = initialPos + doubleForwardOffset;
            if (((doubleForwardPosition & ~63) == 0) && ((1L << doubleForwardPosition) & (playerMask | opponentMask)) == 0 &&
                    ((1L << (initialPos + forwardOffset)) & (playerMask | opponentMask)) == 0) {
                result.add(new Move(piece.clone(), initialPos, doubleForwardPosition, null, Move.CastlingType.NONE, null));
            }
        }

        // Capture moves
        int[] captureOffsets = isWhite ? new int[]{7, 9} : new int[]{-9, -7};
        int[] captureRowOffsets = isWhite ? new int[]{1, 1} : new int[]{-1, -1};
        int[] captureColOffsets = new int[]{-1, 1};
        for (int i = 0; i < captureOffsets.length; i++) {
            int capturePosition = initialPos + captureOffsets[i];
            int newCol = initialCol + captureColOffsets[i];
            int newRow = initialRow + captureRowOffsets[i];
            if (((capturePosition & ~63) == 0) && ((1L << capturePosition) & opponentMask) != 0
                    && (newCol & ~7) == 0 && (newRow & ~7) == 0) {
                handlePawnPromotion(piece, initialPos, capturePosition, capturePosition, board, result);
            } else if (board.getEnPassantActiveSquare() == capturePosition) {
                int capturedPiecePosition = capturePosition + (isWhite ? -8 : 8);
                Piece originalPiece = board.getPiece(opponent, capturedPiecePosition);
                Piece opponentPiece = originalPiece != null ? originalPiece.clone() : null;
                result.add(new Move(piece.clone(), initialPos, capturePosition, opponentPiece, Move.CastlingType.NONE, null));
            }
        }

        return result;
    }

    private void handlePawnPromotion(Piece piece, int initialPos, int newPosition, int capturedPiecePosition, Board board, ArrayList<Move> result) {
        if (capturedPiecePosition < 0) {
            capturedPiecePosition = newPosition;
        }
        Piece originalPiece = board.getPiece(piece.getColor().opponent(), capturedPiecePosition);
        Piece opponentPiece = originalPiece != null ? originalPiece.clone() : null;
        int newRow = newPosition >> 3; // equivalent to / 8, but faster
        boolean promotionRow = (piece.getColor() == Piece.Color.WHITE && newRow == 7) || (piece.getColor() == Piece.Color.BLACK && newRow == 0);

        if (promotionRow) {
            Piece.Type[] promotionPieces = {Piece.Type.QUEEN, Piece.Type.ROOK, Piece.Type.BISHOP, Piece.Type.NIGHT};
            for (Piece.Type promotionPiece : promotionPieces) {
                result.add(new Move(piece.clone(), initialPos, newPosition, opponentPiece,
                        Move.CastlingType.NONE, new Piece(promotionPiece, board.getCurrentPlayer(), newPosition)));
            }
        } else {
            result.add(new Move(piece.clone(), initialPos, newPosition, opponentPiece,
                    Move.CastlingType.NONE, null));
        }
    }


    private ArrayList<Move> generateSlidingMoves(Piece piece, Board board) {
        ArrayList<Move> result = new ArrayList<>();
        Piece.Color currentPlayer = board.getCurrentPlayer();
        Piece.Color opponent = currentPlayer.opponent();
        int initialPos = piece.getPosition();
        long playerMask = board.getPieceMask(currentPlayer);
        long opponentMask = board.getPieceMask(opponent);
        for (Piece.Direction direction : piece.getMoveOptions()) {
            int newPosition = initialPos;
            int newCol = newPosition & 7; // equivalent to % 8, but faster
            int newRow = newPosition >> 3; // equivalent to / 8, but faster

            while (true) {
                newPosition += direction.getOffset();
                newCol += direction.getColOffset();
                newRow += direction.getRowOffset();
                if ((newCol & ~7) != 0 || (newRow & ~7) != 0) {
                    break;
                }

                long newPositionMask = 1L << newPosition;
                if ((newPositionMask & playerMask) != 0) {
                    break;
                }
                Piece originalPiece = board.getPiece(piece.getColor().opponent(), newPosition);
                Piece opponentPiece = originalPiece != null ? originalPiece.clone() : null;
                result.add(new Move(piece.clone(), initialPos, newPosition, opponentPiece,
                        Move.CastlingType.NONE, null));

                if ((newPositionMask & opponentMask) != 0) {
                    break;
                }
            }
        }
        return result;
    }



    // Handles the case where a piece tries to move off the board
    private boolean checkIfOffBoard(int newPosition){
        return newPosition < 0 || newPosition > 63;
    }

    // Handles the case where a piece "teleports" from one side of the board to another
    private boolean checkIfWrapAround(int newPosition, int newCol, int newRow){
        int appropriateCol = newPosition % 8;
        int appropriateRow = newPosition / 8;
        return !(appropriateCol != newCol || appropriateRow != newRow);
    }

    public boolean generateNightMovesCheck(int initialPos, Board board, Piece.Color currentPlayer) {
        // Save the initial position
        int initialCol = initialPos & 7; // equivalent to % 8, but faster
        int initialRow = initialPos >> 3; // equivalent to / 8, but faster
        Piece.Color opponent = currentPlayer.opponent();
        long playerMask = board.getPieceMask(currentPlayer);

        // Check moves in each direction
        for (Piece.Direction direction : Piece.Type.NIGHT.getTypeMoveOptions()) {
            // Updates positions
            int newPosition = initialPos + direction.getOffset();
            int newCol = initialCol + direction.getColOffset();
            int newRow = initialRow + direction.getRowOffset();

            // Wraps around, off-board or blocked by same color.
            if ((newCol & ~7) != 0 || (newRow & ~7) != 0 || ((1L << newPosition) & playerMask) != 0) {
                continue;
            }

            Piece opponentPiece = board.getPiece(opponent, newPosition);
            // Add move
            if (opponentPiece != null && opponentPiece.getType() == Piece.Type.NIGHT) {
                return true;
            }
        }
        return false;
    }

    public boolean generatePawnMovesCheck(int initialPos, Board board, Piece.Color currentPlayer) {

        int initialCol = initialPos & 7; // equivalent to % 8, but faster
        int initialRow = initialPos >> 3; // equivalent to / 8, but faster
        Piece.Color opponent = currentPlayer.opponent();
        long playerMask = board.getPieceMask(currentPlayer);
        long opponentMask = board.getPieceMask(opponent);
        boolean isWhite = currentPlayer.isWhite();

        // Capture moves
        int[] captureOffsets = isWhite ? new int[]{7, 9} : new int[]{-9, -7};
        int[] captureRowOffsets = isWhite ? new int[]{1, 1} : new int[]{-1, -1};
        int[] captureColOffsets = new int[]{-1, 1};
        for (int i = 0; i < captureOffsets.length; i++) {
            int capturePosition = initialPos + captureOffsets[i];
            int newCol = initialCol + captureColOffsets[i];
            int newRow = initialRow + captureRowOffsets[i];
            if ((newCol & ~7) == 0 && (newRow & ~7) == 0 && ((1L << capturePosition) & opponentMask) != 0) {
                Piece opponentPiece = board.getPiece(opponent, capturePosition);
                if (opponentPiece != null && opponentPiece.getType() == Piece.Type.PAWN) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean generateSlidingMovesCheck(int initialPos, Piece.Type slidingPieceType, Board board, Piece.Color currentPlayer) {
        Piece.Color opponent = currentPlayer.opponent();
        long playerMask = board.getPieceMask(currentPlayer);
        long opponentMask = board.getPieceMask(opponent);

        for (Piece.Direction direction : slidingPieceType.getTypeMoveOptions()) {
            int newPosition = initialPos;
            int newCol = newPosition & 7; // equivalent to % 8, but faster
            int newRow = newPosition >> 3; // equivalent to / 8, but faster

            while (true) {
                newPosition += direction.getOffset();
                newCol += direction.getColOffset();
                newRow += direction.getRowOffset();

                if (checkIfOffBoard(newPosition) || !checkIfWrapAround(newPosition, newCol, newRow)) {
                    break;
                }

                long newPositionMask = 1L << newPosition;
                if ((newPositionMask & playerMask) != 0) {
                    break;
                }

                Piece opponentPiece = board.getPiece(opponent, newPosition);
                if (opponentPiece != null && (opponentPiece.getType() == slidingPieceType || opponentPiece.getType() == Piece.Type.QUEEN)) {
                    return true;
                }

                if ((newPositionMask & opponentMask) != 0) {
                    break;
                }
            }
        }
        return false;
    }


}
