"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller.
Parts of this code were originally based on the gtp module 
in the Deep-Go project by Isaac Henrion and Amos Storkey 
at the University of Edinburgh.
"""
import traceback
import time
import numpy as np
import re
from sys import stdin, stdout, stderr, setrecursionlimit
from typing import Any, Callable, Dict, List, Tuple

setrecursionlimit(2000000000)

from board_base import (
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR, GO_POINT,
    PASS,
    MAXSIZE,
    coord_to_point,
    opponent
)
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine

class GtpConnection:
    def __init__(self, go_engine: GoEngine, board: GoBoard, debug_mode: bool = False) -> None:
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board: 
            Represents the current board state.
        """
        self.BlackPriority = {"unknown": 3,"b": 2,"draw" : 1,"w": 0, "N/A": -1}
        self.WhitePriority = {"unknown": 3,"w": 2,"draw" : 1,"b": 0, "N/A": -1}
        self.timelimit = 1
        self.startTime = 0
        
        self._debug_mode: bool = debug_mode
        self.go_engine = go_engine
        self.board: GoBoard = board
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
            "gogui-rules_captured_count": self.gogui_rules_captured_count_cmd,
            "gogui-rules_game_id": self.gogui_rules_game_id_cmd,
            "gogui-rules_board_size": self.gogui_rules_board_size_cmd,
            "gogui-rules_side_to_move": self.gogui_rules_side_to_move_cmd,
            "gogui-rules_board": self.gogui_rules_board_cmd,
            "gogui-analyze_commands": self.gogui_analyze_cmd,
            "timelimit": self.timelimit_cmd,
            "solve": self.solve_cmd
        }

        # argmap is used for argument checking
        # values: (required number of arguments,
        #          error message on argnum failure)
        self.argmap: Dict[str, Tuple[int, str]] = {
            "boardsize": (1, "Usage: boardsize INT"),
            "komi": (1, "Usage: komi FLOAT"),
            "known_command": (1, "Usage: known_command CMD_NAME"),
            "genmove": (1, "Usage: genmove {w,b}"),
            "play": (2, "Usage: play {b,w} MOVE"),
            "legal_moves": (1, "Usage: legal_moves {w,b}"),
        }

    def write(self, data: str) -> None:
        stdout.write(data)

    def flush(self) -> None:
        stdout.flush()

    def start_connection(self) -> None:
        """
        Start a GTP connection. 
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command: str) -> None:
        """
        Parse command string and execute it
        """
        if len(command.strip(" \r\t")) == 0:
            return
        if command[0] == "#":
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements: List[str] = command.split()
        if not elements:
            return
        command_name: str = elements[0]
        args: List[str] = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".format(traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error("Unknown command")
            stdout.flush()

    def has_arg_error(self, cmd: str, argnum: int) -> bool:
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg: str) -> None:
        """ Write msg to the debug stream """
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg: str) -> None:
        """ Send error msg to stdout """
        stdout.write("? {}\n\n".format(error_msg))
        stdout.flush()

    def respond(self, response: str = "") -> None:
        """ Send response to stdout """
        stdout.write("= {}\n\n".format(response))
        stdout.flush()

    def reset(self, size: int) -> None:
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self) -> str:
        return str(GoBoardUtil.get_twoD_board(self.board))

    def protocol_version_cmd(self, args: List[str]) -> None:
        """ Return the GTP protocol version being used (always 2) """
        self.respond("2")

    def quit_cmd(self, args: List[str]) -> None:
        """ Quit game and exit the GTP interface """
        self.respond()
        exit()

    def name_cmd(self, args: List[str]) -> None:
        """ Return the name of the Go engine """
        self.respond(self.go_engine.name)

    def version_cmd(self, args: List[str]) -> None:
        """ Return the version of the  Go engine """
        self.respond(str(self.go_engine.version))

    def clear_board_cmd(self, args: List[str]) -> None:
        """ clear the board """
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args: List[str]) -> None:
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()

    def showboard_cmd(self, args: List[str]) -> None:
        self.respond("\n" + self.board2d())

    def komi_cmd(self, args: List[str]) -> None:
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args: List[str]) -> None:
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args: List[str]) -> None:
        """ list all supported GTP commands """
        self.respond(" ".join(list(self.commands.keys())))

    def legal_moves_cmd(self, args: List[str]) -> None:
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color: str = args[0].lower()
        color: GO_COLOR = color_to_int(board_color)
        moves: List[GO_POINT] = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves: List[str] = []
        for move in moves:
            coords: Tuple[int, int] = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = " ".join(sorted(gtp_moves))
        self.respond(sorted_moves)

    """
    ==========================================================================
    Assignment 2 - game-specific commands start here
    ==========================================================================
    """
    """
    ==========================================================================
    Assignment 2 - commands we already implemented for you
    ==========================================================================
    """
    def gogui_analyze_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        self.respond("pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
                     "pstring/Side to Play/gogui-rules_side_to_move\n"
                     "pstring/Final Result/gogui-rules_final_result\n"
                     "pstring/Board Size/gogui-rules_board_size\n"
                     "pstring/Rules GameID/gogui-rules_game_id\n"
                     "pstring/Show Board/gogui-rules_board\n"
                     )

    def gogui_rules_game_id_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        self.respond("Ninuki")

    def gogui_rules_board_size_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        self.respond(str(self.board.size))

    def gogui_rules_side_to_move_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)

    def gogui_rules_board_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        size = self.board.size
        str = ''
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                #str += '.'
                point = self.board.board[start + i]
                if point == BLACK:
                    str += 'X'
                elif point == WHITE:
                    str += 'O'
                elif point == EMPTY:
                    str += '.'
                else:
                    assert False
            str += '\n'
        self.respond(str)


    def gogui_rules_final_result_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        result1 = self.board.detect_five_in_a_row()
        result2 = EMPTY
        if self.board.get_captures(BLACK) >= 10:
            result2 = BLACK
        elif self.board.get_captures(WHITE) >= 10:
            result2 = WHITE

        if (result1 == BLACK) or (result2 == BLACK):
            self.respond("black")
        elif (result1 == WHITE) or (result2 == WHITE):
            self.respond("white")
        elif self.board.get_empty_points().size == 0:
            self.respond("draw")
        else:
            self.respond("unknown")
        return

    def gogui_rules_legal_moves_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        if (self.board.detect_five_in_a_row() != EMPTY) or \
            (self.board.get_captures(BLACK) >= 10) or \
            (self.board.get_captures(WHITE) >= 10):
            self.respond("")
            return
        legal_moves = self.board.get_empty_points()
        gtp_moves: List[str] = []
        for move in legal_moves:
            coords: Tuple[int, int] = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = " ".join(sorted(gtp_moves))
        self.respond(sorted_moves)

    def play_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        try:
            board_color = args[0].lower()
            board_move = args[1]
            if board_color not in ['b', 'w']:
                self.respond('illegal move: "{} {}" wrong color'.format(board_color, board_move))
                return
            coord = move_to_coord(args[1], self.board.size)
            move = coord_to_point(coord[0], coord[1], self.board.size)
            
            color = color_to_int(board_color)
            if not self.board.play_move(move, color):
                # self.respond("Illegal Move: {}".format(board_move))
                self.respond('illegal move: "{} {}" occupied'.format(board_color, board_move))
                return
            else:
                # self.board.try_captures(coord, color)
                self.debug_msg(
                    "Move: {}\nBoard:\n{}\n".format(board_move, self.board2d())
                )
            if len(args) > 2 and args[2] == 'print_move':
                move_as_string = format_point(coord)
                self.respond(move_as_string.lower())
            else:
                self.respond()
        except Exception as e:
            self.respond('illegal move: "{} {}" {}'.format(args[0], args[1], str(e)))

    def gogui_rules_captured_count_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 2 """
        self.respond(str(self.board.get_captures(WHITE))+' '+str(self.board.get_captures(BLACK)))

    """
    ==========================================================================
    Assignment 2 - game-specific commands you have to implement or modify
    ==========================================================================
    """

    def genmove_cmd(self, args: List[str]) -> None:
        """ 
        Modify this function for Assignment 2.
        """
        board_color = args[0].lower()
        color = color_to_int(board_color)
        result1 = self.board.detect_five_in_a_row()
        result2 = EMPTY
        if self.board.get_captures(opponent(color)) >= 10:
            result2 = opponent(color)
        if result1 == opponent(color) or result2 == opponent(color):
            self.respond("resign")
            return
        legal_moves = self.board.get_empty_points()
        if legal_moves.size == 0:
            self.respond("pass")
            return
        

        self.startTime = time.time()
        
        CurrentBoard = np.copy(self.board.board)
        CurrentWhiteCaptures = self.board.white_captures
        CurrentBlackCaptures = self.board.black_captures
        
        Res = self.Minimax(self.board.current_player, True)
        
        self.board = CurrentBoard
        self.board.white_captures = CurrentWhiteCaptures
        self.board.black_captures = CurrentBlackCaptures

        if Res[0] == board_color or Res[0] == "draw":
            move_coord = point_to_coord(Res[1], self.board.size)
            move_as_string = format_point(move_coord)
            self.play_cmd([board_color, move_as_string, 'print_move'])
        else:
            rng = np.random.default_rng()
            choice = rng.choice(len(legal_moves))
            move = legal_moves[choice]
            move_coord = point_to_coord(move, self.board.size)
            move_as_string = format_point(move_coord)
            self.play_cmd([board_color, move_as_string, 'print_move'])
    
    def timelimit_cmd(self, args: List[str]) -> None:
        if args[0].isnumeric():
            timevar = int(args[0])
            if timevar >= 1 and timevar <= 100:
                self.timelimit = timevar
    
    def who_won(self, Color):
        if time.time() - self.startTime >= self.timelimit:
            return "unknown"
        result1 = self.board.detect_five_in_a_row()
        result2 = EMPTY

        if self.board.get_captures(BLACK) >= 10:
            result2 = BLACK
        elif self.board.get_captures(WHITE) >= 10:
            result2 = WHITE

        if (result1 == BLACK) or (result2 == BLACK):
            return "b"
        elif (result1 == WHITE) or (result2 == WHITE):
            return "w"
        elif self.board.get_empty_points().size == 0:
            return "draw"
        else:
            return False
        
    def get_moves(self,Color):
        CurrentWhiteCaptures = self.board.get_captures(WHITE)
        CurrentBlackCaptures = self.board.get_captures(BLACK)

        Check = None
        if Color == BLACK:
            # This part is for immediate win
            Check = self.board.pattern_check(("EBBBB",[0])) or self.board.pattern_check(("BEBBB",[1])) or self.board.pattern_check(("BBEBB",[2])) or self.board.pattern_check(("BBBEB",[3])) or self.board.pattern_check(("BBBBE",[4]))

            if CurrentBlackCaptures >= 8:
                Check = Check or self.board.pattern_check(("BWWE",[3])) or self.board.pattern_check(("EWWB",[0]))

            
            # This part is for block immediate win
            if CurrentWhiteCaptures >= 8:
                Check = Check or self.board.pattern_check(("WBBE",[3])) or self.board.pattern_check(("EBBW",[0]))
            
            Check = self.board.pattern_check(("EWWWW",[0])) or self.board.pattern_check(("WEWWW",[1])) or self.board.pattern_check(("WWEWW",[2])) or self.board.pattern_check(("WWWEW",[3])) or self.board.pattern_check(("WWWWE",[4]))
            
            # This part is for win in 2 moves
            Check = Check or self.board.pattern_check(("EBBBEE",[4])) or self.board.pattern_check(("EEBBBE",[1]))
            Check = Check or self.board.pattern_check(("EBBEBE",[3])) or self.board.pattern_check(("EBEBBE",[2]))
            
        else:
            # This first part is for immediate win
            Check = self.board.pattern_check(("EWWWW",[0])) or self.board.pattern_check(("WEWWW",[1])) or self.board.pattern_check(("WWEWW",[2])) or self.board.pattern_check(("WWWEW",[3])) or self.board.pattern_check(("WWWWE",[4]))

            if CurrentWhiteCaptures >= 8:
                Check = Check or self.board.pattern_check(("WBBE",[3])) or self.board.pattern_check(("EBBW",[0]))

            
            # This part is for block immediate win
            if CurrentBlackCaptures >= 8:
                Check = Check or self.board.pattern_check(("BWWE",[3])) or self.board.pattern_check(("EWWB",[0]))
            
            Check = self.board.pattern_check(("EBBBB",[0])) or self.board.pattern_check(("BEBBB",[1])) or self.board.pattern_check(("BBEBB",[2])) or self.board.pattern_check(("BBBEB",[3])) or self.board.pattern_check(("BBBBE",[4]))
            
            # This part is for win in 2 moves
            Check = Check or self.board.pattern_check(("EWWWEE",[4])) or self.board.pattern_check(("EEWWWE",[1]))
            Check = Check or self.board.pattern_check(("EWWEWE",[3])) or self.board.pattern_check(("EWEWWE",[2]))

        if Check:
            return Check
        else:
            return self.board.get_empty_points()

    def get_best_value(self,Color,ValOne,ValTwo):
        if Color == WHITE:
            if self.WhitePriority[ValOne] >= self.WhitePriority[ValTwo]:
                return ValOne
            else:
                return ValTwo
        else:
            if self.BlackPriority[ValOne] >= self.BlackPriority[ValTwo]:
                return ValOne
            else:
                return ValTwo
    
    def Minimax(self, Color, ReturnMove = False, Alpha = "N/A", Beta = "N/A"):
        Alpha = Alpha
        Beta = Beta
        Move = PASS
        
        CurrentBoard = np.copy(self.board.board)
        CurrentWhiteCaptures = self.board.white_captures
        CurrentBlackCaptures = self.board.black_captures

        Moves = self.get_moves(Color)
        
        Res = self.who_won(Color)
        if Res:
            #print(Res)
            return Res
        
        if Color == WHITE:
            Val = "N/A" # Lowest Priority
            for i in Moves:
                Move = i

                self.board.board = np.copy(CurrentBoard)
                self.board.white_captures = CurrentWhiteCaptures
                self.board.black_captures = CurrentBlackCaptures

                self.board.play_move(i, Color)
                #time.sleep(0.1)
                #print(self.board2d())
                #print()

                Val = self.get_best_value(Color, Val, self.Minimax(BLACK,False,Alpha,Beta))
                Alpha = self.get_best_value(Color, Alpha, Val)
                if ((Alpha != "N/A" and Beta != "N/A") and (Beta == Alpha or Beta == "b" or Alpha == "w")) or Beta == "unknown" or Alpha == "unknown":
                    #print(Alpha,Beta)
                    break
            
            self.board.board = np.copy(CurrentBoard)
            self.board.white_captures = CurrentWhiteCaptures
            self.board.black_captures = CurrentBlackCaptures
            
            if ReturnMove:
                return (Val, Move)
            else:
                return Val
        else:
            Val = "N/A" # Lowest Priority
            for i in Moves:
                Move = i
                
                self.board.board = np.copy(CurrentBoard)
                self.board.white_captures = CurrentWhiteCaptures
                self.board.black_captures = CurrentBlackCaptures

                self.board.play_move(i, Color)
                #time.sleep(0.1)
                #print(self.board2d())
                #print()

                Val = self.get_best_value(Color, Val, self.Minimax(WHITE,False,Alpha,Beta))
                Beta = self.get_best_value(Color, Beta, Val)
                if ((Alpha != "N/A" and Beta != "N/A") and (Beta == Alpha or Beta == "b" or Alpha == "w")) or Beta == "unknown" or Alpha == "unknown":
                    #print(Alpha,Beta)
                    break
            
            self.board.board = np.copy(CurrentBoard)
            self.board.white_captures = CurrentWhiteCaptures
            self.board.black_captures = CurrentBlackCaptures
            
            if ReturnMove:
                return (Val, Move)
            else:
                return Val
    
    def solve_cmd(self, args: List[str]) -> None:
        Color = self.board.current_player
        self.startTime = time.time()
        
        CurrentBoard = np.copy(self.board.board)
        CurrentWhiteCaptures = self.board.white_captures
        CurrentBlackCaptures = self.board.black_captures

        Res = self.Minimax(self.board.current_player, True)

        self.board.board = CurrentBoard
        self.board.white_captures = CurrentWhiteCaptures
        self.board.black_captures = CurrentBlackCaptures
        
        if (Res[0] == "b" and Color == BLACK) or (Res[0] == "w" and Color == WHITE) or Res[0] == "draw":
            self.respond(Res[0] + " " + str( format_point( point_to_coord(Res[1], self.board.size) ) ).lower() )
        else:
            self.respond(Res[0])

    """
    ==========================================================================
    Assignment 1 - game-specific commands end here
    ==========================================================================
    """

def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
    """
    Transform point given as board array index 
    to (row, col) coordinate representation.
    Special case: PASS is transformed to (PASS,PASS)
    """
    if point == PASS:
        return (PASS, PASS)
    else:
        NS = boardsize + 1
        return divmod(point, NS)


def format_point(move: Tuple[int, int]) -> str:
    """
    Return move coordinates as a string such as 'A1', or 'PASS'.
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if move[0] == PASS:
        return "PASS"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1] + str(row)


def move_to_coord(point_str: str, board_size: int) -> Tuple[int, int]:
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    s = point_str.lower()
    if s == "pass":
        return (PASS, PASS)
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        col = ord(col_c) - ord("a")
        if col_c < "i":
            col += 1
        row = int(s[1:])
        if row < 1:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError("wrong coordinate")
    if not (col <= board_size and row <= board_size):
        raise ValueError("wrong coordinate")
    return row, col


def color_to_int(c: str) -> int:
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK, "w": WHITE, "e": EMPTY, "BORDER": BORDER}
    return color_to_int[c]
