package com.example.chessv2;

import java.io.Serial;
import java.io.Serializable;
import java.security.SecureRandom;

public class Zobrist implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;
    private static final int BOARD_SIZE = 64;
    private static final int PIECE_TYPES = 12;
    private static final int PLAYER_TYPES = 2;

    private final long[][][] randomValues;

    public Zobrist() {
        SecureRandom random = new SecureRandom();
        randomValues = new long[BOARD_SIZE][PIECE_TYPES][PLAYER_TYPES];

        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < PIECE_TYPES; j++) {
                for (int k = 0; k < PLAYER_TYPES; k++) {
                    randomValues[i][j][k] = random.nextLong();
                }
            }
        }
    }

    public long generateHash(long[] chessPieces, Piece.Color player) {
        long hash = 0L;

        for (int i = 0; i < BOARD_SIZE; i++) {
            for(int j = 0; j < chessPieces.length; j++){
                if (((chessPieces[j] >> i) & 1) == 1) {
                    hash ^= randomValues[i][j][0];
                    break;
                }
            }
            if (player == Piece.Color.WHITE) {
                hash ^= randomValues[i][0][1];
            } else {
                hash ^= randomValues[i][0][0];
            }
        }

        return hash;
    }
}

