package com.example.chessv2;

public class Piece {

    public static final Type[] indexToType = new Type[]{Type.PAWN, Type.NIGHT, Type.BISHOP, Type.ROOK, Type.QUEEN, Type.KING,
            Type.PAWN, Type.NIGHT, Type.BISHOP, Type.ROOK, Type.QUEEN, Type.KING};

    enum Direction {
        LOWER_LEFT(-17, -2, -1),
        LOW_LEFT(-10, -1, -2),
        UP_LEFT(6, 1, -2),
        UPPER_LEFT(15, 2, -1),
        UPPER_RIGHT(17, 2, 1),
        UP_RIGHT(10, 1, 2),
        LOW_RIGHT(-6, -1, 2),
        LOWER_RIGHT(-15, -2, 1),
        WHITE_MOVE_NORTH(8, 1, 0),
        WHITE_DOUBLE_MOVE_NORTH(16, 2, 0),
        WHITE_CAPTURE_NORTHWEST(7, 1, -1),
        WHITE_CAPTURE_NORTHEAST(9, 1, 1),
        BLACK_MOVE_SOUTH(-8, -1, 0),
        BLACK_DOUBLE_MOVE_SOUTH(-16, -2, 0),
        BLACK_CAPTURE_SOUTHWEST(-9, -1, -1),
        BLACK_CAPTURE_SOUTHEAST(-7, -1, 1),
        NORTHEAST(9, 1, 1),
        NORTHWEST(7, 1, -1),
        SOUTHEAST(-7, -1, 1),
        SOUTHWEST(-9, -1, -1),
        NORTH(8, 1, 0),
        EAST(1, 0, 1),
        SOUTH(-8, -1, 0),
        WEST(-1, 0, -1);

        private final int offset;
        private final int rowOffset;
        private final int colOffset;

        Direction(int offset, int rowOffset, int colOffset) {
            this.offset = offset;
            this.rowOffset = rowOffset;
            this.colOffset = colOffset;
        }

        public int getOffset() {
            return offset;
        }

        public int getRowOffset() {
            return rowOffset;
        }

        public int getColOffset() {
            return colOffset;
        }

        public boolean isPawnCapture() {
            return (this == BLACK_CAPTURE_SOUTHEAST || this == BLACK_CAPTURE_SOUTHWEST
                    || this == WHITE_CAPTURE_NORTHEAST || this == WHITE_CAPTURE_NORTHWEST);
        }

        public boolean isPawnDoubleMove() {
            return (this == WHITE_DOUBLE_MOVE_NORTH || this == BLACK_DOUBLE_MOVE_SOUTH);
        }

        public int halfwayOffset() {
            return (this == WHITE_DOUBLE_MOVE_NORTH ? WHITE_MOVE_NORTH.getOffset() : BLACK_MOVE_SOUTH.getOffset());
        }

        public boolean isCorrectColorPawnMove(Color color) {
            if (color == Color.WHITE) {
                return (this == WHITE_MOVE_NORTH || this == WHITE_DOUBLE_MOVE_NORTH || this == WHITE_CAPTURE_NORTHWEST || this == WHITE_CAPTURE_NORTHEAST);
            }
            return (this == BLACK_MOVE_SOUTH || this == BLACK_DOUBLE_MOVE_SOUTH || this == BLACK_CAPTURE_SOUTHWEST || this == BLACK_CAPTURE_SOUTHEAST);
        }
    }

    // Flags the piece with a type based on the index.
    enum Type {
        PAWN(new Direction[]{Direction.WHITE_CAPTURE_NORTHEAST,
                Direction.WHITE_CAPTURE_NORTHEAST,
                Direction.WHITE_MOVE_NORTH,
                Direction.WHITE_DOUBLE_MOVE_NORTH,
                Direction.BLACK_CAPTURE_SOUTHEAST,
                Direction.BLACK_CAPTURE_SOUTHWEST,
                Direction.BLACK_MOVE_SOUTH,
                Direction.BLACK_DOUBLE_MOVE_SOUTH,},
                100),
        ROOK(new Direction[]{Direction.NORTH,
                Direction.EAST,
                Direction.SOUTH,
                Direction.WEST},
                500),
        NIGHT(new Direction[]{Direction.LOWER_LEFT,
                Direction.LOW_LEFT,
                Direction.UP_LEFT,
                Direction.UPPER_LEFT,
                Direction.UPPER_RIGHT,
                Direction.UP_RIGHT,
                Direction.LOW_RIGHT,
                Direction.LOWER_RIGHT},
                320),
        BISHOP(new Direction[]{Direction.NORTHEAST,
                Direction.NORTHWEST,
                Direction.SOUTHWEST,
                Direction.SOUTHEAST},
                330),
        QUEEN(new Direction[]{Direction.NORTHEAST,
                Direction.NORTHWEST,
                Direction.SOUTHWEST,
                Direction.SOUTHEAST,
                Direction.NORTH,
                Direction.EAST,
                Direction.SOUTH,
                Direction.WEST},
                900),
        KING(new Direction[]{Direction.NORTHEAST,
                Direction.NORTHWEST,
                Direction.SOUTHWEST,
                Direction.SOUTHEAST,
                Direction.NORTH,
                Direction.EAST,
                Direction.SOUTH,
                Direction.WEST},
                20000);

        private final Direction[] moveOptions;
        private final int pieceValue;

        Type(Direction[] moveOptions, int pieceValue) {
            this.moveOptions = moveOptions;
            this.pieceValue = pieceValue;
        }

        public Direction[] getTypeMoveOptions() {
            return moveOptions;
        }

        public int getPieceValue() {
            return pieceValue;
        }
    }


    enum Color {
        BLACK,
        WHITE;

        public Color opponent() {
            if (this == WHITE) {
                return BLACK;
            } else return WHITE;
        }

        public boolean isWhite() {
            return this == WHITE;
        }
    }

    private Color color;

    private Type type;

    private int position;


    public Piece(int index, int position) {
        // Set current position
        this.position = position;
        // Determine Piece Color
        color = index / 6 >= 1 ? Color.BLACK : Color.WHITE;
        // Determine Piece Type
        type = indexToType[index];
    }

    public Piece(Piece.Type type, Piece.Color color, int position) {
        // Set current position
        this.position = position;
        // Determine Piece Color
        this.color = color;
        // Determine Piece Type
        this.type = type;
    }

    public int getPosition() {
        return position;
    }

    public void setPosition(int position) {
        this.position = position;
    }


    public Color getColor() {
        return color;
    }

    public Type getType() {
        return type;
    }

    public void setType(Piece.Type type) {
        this.type = type;
    }

    public Direction[] getMoveOptions() {
        return type.moveOptions;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder(this.type.toString().charAt(0));
        builder.append((char) ('a' + position % 8));
        builder.append((char) ('1' + position / 8));
        return builder.toString();
    }

    @Override
    public Piece clone() {
        return new Piece(this.type, this.color, this.position);
    }

}
