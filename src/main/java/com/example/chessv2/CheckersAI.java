package com.example.chessv2;


// As of 12:27 Sunday, March 5th, there are three glaring errors.

// SOLVED - the double jump functionality allows for a jump to be split if it hits
// the point where another jump is of equal length, need to lock in the piece
// that can move, might do this by creating a new ChessBoard constructor that
// takes in the last piece move and when generating moves only considers that piece.

// SOLVED - Regarding the above paragraph, I fixed the inability to lock in place paths by modifying the move object. However, this undid
// My work solving the issue of kings being able to jump a piece immediately on becoming a king, so I need to do that next
// Otherwise, there is a new issue this fix introduced, where the player can't actually double jump, so yeah that is fun.


// SOLVED - for some reason, if the enemy can jump a piece and king (or maybe just king in general), the
// computer completely overlooks this option.

// SOLVED - for some reason, potentially related, the evaluation can pass 73,000

// Maybe I fixed that /^_^\


// To-do List:
// 1. COMPLETE - Fix errors above
// 2. COMPLETE - Reimplement transpositionTable that stores several of the previous best moves
// 3. COMPLETE - Highlight the pieces that have legal moves before they are clicked.
// 4. COMPLETE - Add a reset button
// 5. COMPLETE - Add animations
// 5.1. Adding animations broke other things
// 5.2. COMPLETE - Red needs proper animations,
// 5.3. COMPLETE - Image view nodes are deleted when a piece passes over it.
// 5.4 Animation needs to repeat for double jumps.
// 5.5 ChessBoard needs to display properly at the end.
// 6. COMPLETE - Add a move counter
// 7. Add an AI VS AI mode.
// 8. KIND OF COMPLETE - Add the ability for the window to resize.
// 9. Neural net.

/*
public class CheckersAI {
    private static final int MAX_AGE = 2;
    private int MAX_DEPTH = 9999;
    private Controller controller;
    // I think I found an error, the xobrist hashing values are different for each new ChessBoard object potentially?
    // Lets a go

    // Pseudo-Code
    /*
     * If depth == 0 or the game state is not ongoing, get the evaluation of the ChessBoard.
     * Generate zobrist hash for position
     * Look in transposition table, if position is present, return that evaluation and best move
     * Otherwise, we need to evaluate the position
     * If player = Black, the maximizing player set Max to IntegerMin value
     * for each piece in .getLegalMoves
     * for each move in legalMoves(piece)
     * make a new ChessBoard with the new move made.
     * recursively call the function
     * if the returned eval is greater than max eval, set it as new max and set the move as the new move
     * Add position to transposition table with these values, though this step might need to come the move before
     * Set alpha value to the new max value
     * if beta is less than or equal to alpha, exit the move loop and return max eval and best move
     *
     * If player = red, the min player set min to IntegerMax value
     * for each piece in .getLegalMoves
     * for each move in legalMoves(piece)
     * make a new ChessBoard with the new move made.
     * recursively call the function
     * if the returned eval is less than min eval, set it as new min and set the move as the new move
     * Add position to transposition table with these values, though this step might need to come the move before
     * Set beta value to the min max value
     * if beta is greater than or equal to alpha, exit the move loop and return min eval and best move
     */

/*
    private final ChessBoard.Player computerPlayer;

    private final HashMap<Long, TranspositionEntry> transpositionTable = new HashMap<>();

    private long startTime;

    private int allowedTime;

    public CheckersAI(ChessBoard.Player player, Controller controller, int time) {
        computerPlayer = player;
        this.controller = controller;
    }

    // Analyzes the ChessBoard and determines the best move, taking advantage of iterative deepening.
    public Move getBestMove(ChessBoard ChessBoard, int time) {
        // Set the time limit for looking for a move.
        allowedTime = time * 1000;
        startTime = System.currentTimeMillis();
        // Purge old entries from the transposition table
        cleanTable();
        // Initialize variables
        int currentDepth = 1;
        boolean timeLimit = false;
        Move result = null;
        int eval = 0;
        try {
            // While the time limit has not been reached, continue running alpha beta at increasing depths
            while (!timeLimit) {
                // Check if the time limit has run out
                timeLimit = outOfTime();
                // Runs a minimax search for the current depth and saves the value
                // I am unsure of this, but it may be worthwhile remembering the previous alpha and beta values
                //TreeMap<Integer, Move> output = minMaxWithoutAlphaBeta(ChessBoard, currentDepth);
                TreeMap<Integer, Move> output = alphaBeta(ChessBoard, Integer.MIN_VALUE, Integer.MAX_VALUE, currentDepth);
                eval = output.firstKey();
                result = output.get(eval);
                // Increase the depth of the game
                currentDepth++;
            }
        } catch (TimeLimitExceededException ignored) {
        }
        updateStats((double) (System.currentTimeMillis() - startTime), transpositionTable.size(), currentDepth, result, eval, evalToPercentage(eval));
        return result;
    }

    public TreeMap<Integer, Move> alphaBeta(ChessBoard chessBoard, int alpha, int beta, int depth) throws TimeLimitExceededException {
        // At this point, it would be prudent to check if a base case has been reached
        // which is a depth of zero or the game finishing
        if(depth == 0 || chessBoard.gameState() != ChessBoard.State.ONGOING){
            return moveToMap(chessBoard.evaluateGame(), null);
        }
        // Generate a hash of the current ChessBoard position
        long hash = chessBoard.generateHash();
        TranspositionEntry entry = transpositionTable.get(hash);
        // Placeholder for a list of legalMoves to be generated later
        LinkedList<Move> bestMoveOrder;
        // Check if the position has been reached before
        if(entry != null){
            // Entry has already been explored
            if(entry.getDepth() >= depth){
                return moveToMap(entry.getEvaluation() + depth, entry.getBestMove());
            } else{
                // Since the position is partially explored, we should utilize the best moves for it
                bestMoveOrder = entry.getBestMoveList();
                mergeLinkedList(bestMoveOrder, chessBoard.getLegalMoves());
            }
        } else{
            // Since the position is brand new, instead of searching blindly
            // uses a function that sorts moves by how likely they are to be good
            bestMoveOrder = chessBoard.getLegalMoves();
        }
        // Determine if maximizing or minimizing
        boolean maximizingPlayer = isMaximizingPlayer(chessBoard);
        // Keeps track of the moves made for the transposition entry.
        TreeMap<Integer, Move> newMoveOrder = new TreeMap<>();
        Move bestMove = null;
        int bestEvaluation = maximizingPlayer ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        // Play each possible move on the ChessBoard and search it out to the end of our game tree
        for(Move move : bestMoveOrder){
            // Create a new ChessBoard to avoid editing the old one.
            ChessBoard tempChessBoard = chessBoard.clone();
            tempChessBoard.move(move);
            // Use recursion to evaluate new ChessBoard
            TreeMap<Integer, Move> tempResult = alphaBeta(tempChessBoard, alpha, beta, depth - 1);
            int tempEvaluation = tempResult.firstKey();
            newMoveOrder.put(tempEvaluation, move);
            if(maximizingPlayer){
                if(tempEvaluation > bestEvaluation){
                    bestEvaluation = tempEvaluation;
                    bestMove = move;
                }
                alpha = Math.max(alpha, bestEvaluation);
            } else{
                if(tempEvaluation < bestEvaluation){
                    bestEvaluation = tempEvaluation;
                    bestMove = move;
                }
                beta = Math.min(beta, bestEvaluation);
            }
            if (beta <= alpha) {
                break;
            }
        }
        if(outOfTime()){
            throw new TimeLimitExceededException();
        }
        transpositionTable.put(hash, new TranspositionEntry(depth, bestEvaluation, 0, newMoveOrder, maximizingPlayer, bestMove));
        return moveToMap(bestEvaluation, bestMove);
    }

    public TreeMap<Integer, Move> minMaxWithoutAlphaBeta(ChessBoard chessBoard, int depth){
        // Generate a hash of the current ChessBoard position
        long hash = chessBoard.generateHash();
        TranspositionEntry entry = transpositionTable.get(hash);
        // Placeholder for a list of legalMoves to be generated later
        LinkedList<Move> bestMoveOrder;
        // Check if the position has been reached before
        if(entry != null){
            // Entry has already been explored
            if(entry.getDepth() >= depth){
                return moveToMap(entry.getEvaluation(), entry.getBestMove());
            } else{
                // Since the position is partially explored, we should utilize the best moves for it
                bestMoveOrder = entry.getBestMoveList();
            }
        } else{
            // Since the position is brand new, instead of searching blindly
            // uses a function that sorts moves by how likely they are to be good
            bestMoveOrder = chessBoard.getLegalMoves();
        }
        // At this point, it would be prudent to check if a base case has been reached
        // which is a depth of zero or the game finishing
        if(depth == 0 || chessBoard.gameState() != ChessBoard.State.ONGOING){
            return moveToMap(chessBoard.evaluateGame(), null);
        }
        // Determine if maximizing or minimizing
        boolean maximizingPlayer = isMaximizingPlayer(chessBoard);
        // Keeps track of the moves made for the transposition entry.
        TreeMap<Integer, Move> newMoveOrder = new TreeMap<>();
        Move bestMove = null;
        int bestEvaluation = maximizingPlayer ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        // Play each possible move on the ChessBoard and search it out to the end of our game tree
        for(Move move : bestMoveOrder){
            // Create a new ChessBoard to avoid editing the old one.
            ChessBoard tempChessBoard = chessBoard.clone();
            tempChessBoard.move(move);
            // Use recursion to evaluate new ChessBoard
            TreeMap<Integer, Move> tempResult = minMaxWithoutAlphaBeta(tempChessBoard, depth - 1);
            int tempEvaluation = tempResult.firstKey();
            newMoveOrder.put(tempEvaluation, move);
            if(maximizingPlayer){
                if(tempEvaluation > bestEvaluation){
                    bestEvaluation = tempEvaluation;
                    bestMove = move;
                }
            } else{
                if(tempEvaluation < bestEvaluation){
                    bestEvaluation = tempEvaluation;
                    bestMove = move;
                }
            }
        }
        transpositionTable.put(hash, new TranspositionEntry(depth, bestEvaluation, 0, newMoveOrder, maximizingPlayer, bestMove));
        return moveToMap(bestEvaluation, bestMove);
    }

    public TreeMap<Integer, Move> minMaxWithoutOrdering(ChessBoard chessBoard, int alpha, int beta, int depth){
        // Generate a hash of the current ChessBoard position
        long hash = chessBoard.generateHash();
        TranspositionEntry entry = transpositionTable.get(hash);
        // Placeholder for a list of legalMoves to be generated later
        // Check if the position has been reached before
        if(entry != null){
            // Entry has already been explored
            if(entry.getDepth() >= depth){
                return moveToMap(entry.getEvaluation(), entry.getBestMove());
            }
        }
        LinkedList<Move> bestMoveOrder = chessBoard.getLegalMoves();
        // At this point, it would be prudent to check if a base case has been reached
        // which is a depth of zero or the game finishing
        if(depth == 0 || chessBoard.gameState() != ChessBoard.State.ONGOING){
            return moveToMap(chessBoard.evaluateGame(), null);
        }
        // Determine if maximizing or minimizing
        boolean maximizingPlayer = isMaximizingPlayer(chessBoard);
        // Keeps track of the moves made for the transposition entry.
        Move bestMove = null;
        int bestEvaluation = maximizingPlayer ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        // Play each possible move on the ChessBoard and search it out to the end of our game tree
        for(Move move : bestMoveOrder){
            // Create a new ChessBoard to avoid editing the old one.
            ChessBoard tempChessBoard = chessBoard.clone();
            tempChessBoard.move(move);
            // Use recursion to evaluate new ChessBoard
            TreeMap<Integer, Move> tempResult = minMaxWithoutOrdering(tempChessBoard, alpha, beta, depth - 1);
            int tempEvaluation = tempResult.firstKey();
            if(maximizingPlayer){
                if(tempEvaluation > bestEvaluation){
                    bestEvaluation = tempEvaluation;
                    bestMove = move;
                }
                alpha = Math.max(alpha, bestEvaluation);
            } else{
                if(tempEvaluation < bestEvaluation){
                    bestEvaluation = tempEvaluation;
                    bestMove = move;
                }
                beta = Math.min(beta, bestEvaluation);
            }
            if (beta <= alpha) {
                break;
            }
        }
        transpositionTable.put(hash, new TranspositionEntry(depth, bestEvaluation, 0, maximizingPlayer, bestMove));
        return moveToMap(bestEvaluation, bestMove);
    }

    // Takes the eval and bestMove and returns them as a tree map
    private TreeMap<Integer, Move> moveToMap(Integer evaluation, Move bestMove) {
        TreeMap<Integer, Move> result = new TreeMap<>();
        result.put(evaluation, bestMove);
        return result;
    }

    // Checks if the time has run out
    private boolean outOfTime() {
        long elapsedTime = System.currentTimeMillis() - startTime;
        return elapsedTime >= allowedTime;
    }
    private void cleanTable() {
            Iterator<Map.Entry<Long, TranspositionEntry>> hmIterator = transpositionTable.entrySet().iterator();
            while (hmIterator.hasNext()) {
                TranspositionEntry entry = hmIterator.next().getValue();
                if (entry.getAge() < MAX_AGE) {
                    entry.age++; // increase the age of the entry
                } else {
                    hmIterator.remove(); // remove the entry if it has reached the max age
                }
            }
    }


    private void updateStats(double time, int gamesSearched, int depth, Move bestMove, Integer eval, double evalBar){
        //controller.setUpdateStats(time, gamesSearched, depth, bestMove, eval, evalBar);
    }

    private double evalToPercentage(Integer firstKey) {
        double result = (double) (firstKey+(1200)) / 2400;
        result = result < 0 ? 0 : result;
        result = result > 1 ? 1 : result;
        return result;
    }

    private TreeMap<Integer, ArrayList<Move>> createEmptyKiller(int maxDepth) {
        TreeMap<Integer, ArrayList<Move>> result = new TreeMap<>();
        for (int i = 1; i <= maxDepth; i++) {
            result.put(i, new ArrayList<>());
        }
        return result;
    }


    private boolean isMaximizingPlayer(ChessBoard chessBoard) {
        return chessBoard.getCurrentPlayer() == computerPlayer;
    }

    // Combines two linked list so that their move ordering is more beneficial.
    private void mergeLinkedList(LinkedList<Move> list1, LinkedList<Move> list2) {
        for (Move value1 : list1) {
            Iterator<Move> iter2 = list2.iterator();
            boolean foundMatch = false;
            while (iter2.hasNext()) {
                Move value2 = iter2.next();
                if (value1.equals(value2)) {
                    iter2.remove();
                    foundMatch = true;
                }
            }
            if (!foundMatch) {
                list2.add(value1);
            }
        }
        list1.addAll(list2);
    }

}

class TimeLimitExceededException extends Exception { }

 */
