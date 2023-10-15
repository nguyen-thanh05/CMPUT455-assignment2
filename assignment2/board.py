"""
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
from typing import List

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    PASS,
    GO_COLOR,
    GO_POINT,
)


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""


class GoBoard(object):
    def __init__(self, size: int) -> None:
        """
        Creates a Go board of given size
        """
        self.board = None
        self.maxpoint = None
        self.current_player = None
        self.last2_move = None
        self.last_move = None
        self.ko_recapture = None
        self.diags = None
        self.WE = None
        self.NS = None
        self.size = None
        self.cols = None
        self.rows = None
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0
        #self.string_rep = None
        #self.__repr__()
        self.removed = None
        self.stack = []
    def __repr__(self):
        """
        Returns a number representation of the board with "w", "b", and "."
        """
        size: int = self.size
        board2d: np.ndarray[GO_POINT] = np.zeros((size, size), dtype=GO_POINT)
        for row in range(size):
            start: int = self.row_start(row + 1)
            board2d[row, :] = self.board[start: start + size]
        board2d = np.flipud(board2d)
        return board2d

    def add_two_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            self.black_captures += 2
        elif color == WHITE:
            self.white_captures += 2

    def get_captures(self, color: GO_COLOR) -> int:
        if color == BLACK:
            return self.black_captures
        elif color == WHITE:
            return self.white_captures

    def calculate_rows_cols_diags(self) -> None:
        if self.size < 5:
            return
        # precalculate all rows, cols, and diags for 5-in-a-row detection
        self.rows = []
        self.cols = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)

            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)

        self.diags = []
        # diag towards SE, starting from first row (1,1) moving right to (1,n)
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_se = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_se.append(pt)
                pt += self.NS + 1
            if len(diag_se) >= 5:
                self.diags.append(diag_se)
        # diag towards SE and NE, starting from (2,1) downwards to (n,1)
        for i in range(start + self.NS, self.row_start(self.size) + 1, self.NS):
            diag_se = []
            diag_ne = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_se.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_ne.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_se) >= 5:
                self.diags.append(diag_se)
            if len(diag_ne) >= 5:
                self.diags.append(diag_ne)
        # diag towards NE, starting from (n,2) moving right to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_ne = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_ne.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_ne) >= 5:
                self.diags.append(diag_ne)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 5) + 1) * 2

    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.ko_recapture: GO_POINT = NO_POINT
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0
        #self.string_rep = None
        #self.__repr__()
        self.removed = None

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        b.black_captures = self.black_captures
        b.white_captures = self.white_captures
        #b.string_rep = None
        #b.__repr__()
        b.removed = self.removed
        return b

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def _is_legal_check_simple_cases(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check the simple cases of illegal moves.
        Some "really bad" arguments will just trigger an assertion.
        If this function returns False: move is definitely illegal
        If this function returns True: still need to check more
        complicated cases such as suicide.
        """
        assert is_black_white(color)
        if point == PASS:
            return True
        # Could just return False for out-of-bounds, 
        # but it is better to know if this is called with an illegal point
        assert self.pt(1, 1) <= point <= self.pt(self.size, self.size)
        assert is_black_white_empty(self.board[point])
        if self.board[point] != EMPTY:
            return False
        if point == self.ko_recapture:
            return False
        return True

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        if point == PASS:
            return True
        board_copy: GoBoard = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def end_of_game(self) -> bool:
        return self.last_move == PASS \
            and self.last2_move == PASS

    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start: start + self.size] = EMPTY

    def is_eye(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block: np.ndarray) -> bool:
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone: GO_POINT) -> np.ndarray:
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color: GO_COLOR = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point: GO_POINT) -> np.ndarray:
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=np.bool_)
        pointstack = [point]
        color: GO_COLOR = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point: GO_POINT) -> GO_POINT:
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns NO_POINT otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture: GO_POINT = NO_POINT
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            self.string_rep[captures] = EMPTY * (10 ** captures)
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture

    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Tries to play a move of color on the point.
        Returns whether the point was empty.
        """
        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        #self.string_rep += int(color) * (4 ** int(point))  # Newly added
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        opp = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS + 1, -(self.NS + 1), self.NS - 1, -self.NS + 1]
        self.removed = []
        for offset in offsets:
            if self.board[point + offset] == opp and self.board[point + (offset * 2)] == opp and self.board[
                    point + (offset * 3)] == color:
                self.removed.append((self.board[point + offset], point + offset, point + (offset * 2)))
                self.board[point + offset] = EMPTY
                self.board[point + (offset * 2)] = EMPTY

                #self.string_rep -= int(self.removed[0]) * (4 ** int(point + offset))  # Newly added
                #self.string_rep -= int(self.removed[0]) * (4 ** int(point + (offset * 2)))  # Newly added

                if color == BLACK:
                    self.black_captures += 2
                else:
                    self.white_captures += 2

        self.stack.append([point, self.removed])
        return True

    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR) -> List:
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT) -> List:
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self) -> List:
        """
        Get the list_to_check of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        if self.last_move != NO_POINT and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != NO_POINT and self.last2_move != PASS:
            board_moves.append(self.last2_move)
        return board_moves

    def detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        for r in self.rows:
            result = self.has_five_in_list(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_list(c)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_five_in_list(d)
            if result != EMPTY:
                return result
        return EMPTY

    def has_five_in_list(self, pos_array):
        """
        Returns BLACK or WHITE if any five in a rows exist in the list_to_check.
        EMPTY otherwise.
        """
        sum = 0
        for i in range(len(pos_array) - 5 + 1):
            if i == 0:
                for j in range(5):
                    sum += self.board[pos_array[j]]
            else:
                sum -= self.board[pos_array[i - 1]]
                sum += self.board[pos_array[i - 1 + 5]]

            if sum == 5:
                return BLACK
            elif sum == -5:
                return WHITE
        return EMPTY

    # Pattern is a string, "E" = Empty, "R" = Return vals, '.' = Placeholder, "B" = Black, "W" = White, "C" = color

    def pattern_check(self, colour):
        return_array = []
        for r in self.rows:
            result = self.pattern_check_list(r, colour)
            if result:
                return_array += result

        for c in self.cols:
            result = self.pattern_check_list(c, colour)
            if result:
                return_array += result

        for d in self.diags:
            result = self.pattern_check_list(d, colour)
            if result:
                return_array += result

        return return_array if len(return_array) > 0 else None

    # Possibly change to KMP pattern matching
    def pattern_check_list(self, pos_array, colour):
        THREAT_THRESHOLD = 4 if colour == BLACK else -4
        sum = 0
        return_array = []
        for i in range(len(pos_array) - 5 + 1):
            if i == 0:
                for j in range(5):
                    sum += self.board[pos_array[j]]
            else:
                sum -= self.board[pos_array[i - 1]]
                sum += self.board[pos_array[i - 1 + 5]]

            if sum == THREAT_THRESHOLD:
                for j in range(i, i + 5):
                    if self.board[pos_array[j]] == EMPTY:
                        return_array.append(pos_array[j])
        return return_array if len(return_array) > 0 else None
        """THREAT_THRESHOLD = 4 if colour == BLACK else -4
        return_array = []

        array_to_check = self.board[pos_array]
        divided_arr = np.lib.stride_tricks.sliding_window_view(array_to_check, 5)  # NEW IN NP 1.20. I double checked on lab machine, they have np 1.20.2 so it should work.
        sum = np.sum(divided_arr, axis=1)
        index = np.where(sum == THREAT_THRESHOLD)[0]

        if len(index) > 0:
            for i in index:
                return_array.append(pos_array[i + np.where(divided_arr[i] == EMPTY)[0][0]])
        return return_array if len(return_array) > 0 else None"""

    def undo(self):
        """    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:

        Tries to play a move of color on the point.
        Returns whether the point was empty.

        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        #self.string_rep += int(color) * (4 ** int(point))  # Newly added
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        opp = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS + 1, -(self.NS + 1), self.NS - 1, -self.NS + 1]
        for offset in offsets:
            if self.board[point + offset] == opp and self.board[point + (offset * 2)] == opp and self.board[
                    point + (offset * 3)] == color:
                self.removed = (self.board[point + offset], point + offset, point + (offset * 2))

                self.board[point + offset] = EMPTY
                self.board[point + (offset * 2)] = EMPTY

                #self.string_rep -= int(self.removed[0]) * (4 ** int(point + offset))  # Newly added
                #self.string_rep -= int(self.removed[0]) * (4 ** int(point + (offset * 2)))  # Newly added

                if color == BLACK:
                    self.black_captures += 2
                else:
                    self.white_captures += 2
        return True"""

        last_move = self.stack.pop(-1)
        point = last_move[0]
        captured = last_move[1]
        colour = self.board[point]
        self.last_move = self.stack[-2][0] if len(self.stack) >= 2 else None
        self.last2_move = self.stack[-3][0] if len(self.stack) >= 3 else None
        if len(captured) > 0:
            for capture in captured:
                self.board[capture[1]] = capture[0]
                self.board[capture[2]] = capture[0]
                if colour == BLACK:
                    self.black_captures -= 2
                elif colour == WHITE:
                    self.white_captures -= 2

        self.board[point] = EMPTY



