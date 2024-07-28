import numpy as np

class EnvironmentMinMax:
    def __init__(self, num_columns, num_rows, num_win) -> None:
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.num_win = num_win
        self.board = np.zeros(num_columns * num_rows)

    def reset(self) -> tuple:
        self.board = np.zeros(self.num_columns * self.num_rows)
        return self.board

    def __check_action(self, action) -> bool:
        return self.board[self.num_rows * self.num_columns - (self.num_columns - action)] == 0

    def __update_board(self, action, player) -> None:
        for i in range(action, self.num_columns * self.num_rows, self.num_columns):
            if self.board[i] == 0:
                self.board[i] = player
                break

    def __check_win(self, player) -> bool:
        # Check rows
        for i in range(self.num_columns * self.num_rows):
            if self.board[i] == player:
                count = 1
                for j in range(1, self.num_win):
                    if i + j * self.num_columns < self.num_columns * self.num_rows and self.board[i + j * self.num_columns] == player:
                        count += 1
                if count == self.num_win:
                    return True

        # Check columns
        for i in range(self.num_columns * self.num_rows):
            if self.board[i] == player:
                count = 1
                for j in range(1, self.num_win):
                    if i + j < self.num_columns * self.num_rows and self.board[i + j] == player:
                        count += 1
                if count == self.num_win:
                    return True

        # Check diagonals
        for i in range(self.num_columns * self.num_rows):
            if self.board[i] == player:
                count = 1
                for j in range(1, self.num_win):
                    if i + j * (self.num_columns + 1) < self.num_columns * self.num_rows and self.board[i + j * (self.num_columns + 1)] == player:
                        count += 1
                if count == self.num_win:
                    return True

        for i in range(self.num_columns * self.num_rows):
            if self.board[i] == player:
                count = 1
                for j in range(1, self.num_win):
                    if i + j * (self.num_columns - 1) < self.num_columns * self.num_rows and self.board[i + j * (self.num_columns - 1)] == player:
                        count += 1
                if count == self.num_win:
                    return True

        return False
                
    def __get_action(self) -> int:
        col, _ = self.minimax(5, -np.inf, np.inf, True)
        return col
    
    def step(self, action) -> tuple:
        # Check if the action is valid
        if self.__check_action(action):
            # Update the board
            self.__update_board(action, 1)
            # Check if the game is over
            if self.__check_win(1):
                return self.board, 1, True
            elif np.all(self.board != 0):
                return self.board, 0, True
            else:
                # Do epsilon greedy move based on Q
                action = self.__get_action()
                self.__update_board(action, -1)
                # Check if the game is over
                if self.__check_win(-1):
                    return self.board, -1, True
                elif np.all(self.board != 0):
                    return self.board, 0, True
                else:
                    # Game is not over
                    return self.board, 0, False
        else:
            return self.board, -100, True
        
    def render(self) -> None:
        for row in range(self.num_rows - 1, -1, -1):
            print("|", end="")
            for col in range(self.num_columns):
                if self.board[row * self.num_columns + col] == 1:
                    print(" X ", end="|")
                elif self.board[row * self.num_columns + col] == -1:
                    print(" O ", end="|")
                else:
                    print("   ", end="|")
            print()
        print("-" * (self.num_columns * 4 + 1))


    # New methods for Minimax implementation
    def get_valid_locations(self):
        return [c for c in range(self.num_columns) if self.__check_action(c)]

    def is_terminal_node(self):
        return self.__check_win(1) or self.__check_win(-1) or len(self.get_valid_locations()) == 0

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = 1 if piece == -1 else -1

        if window.count(piece) == self.num_win:
            score += 100
        elif window.count(piece) == self.num_win - 1 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == self.num_win - 2 and window.count(0) == 2:
            score += 2

        if window.count(opp_piece) == self.num_win - 1 and window.count(0) == 1:
            score -= 4

        return score

    def score_position(self, piece):
        score = 0

        # Score center column
        center_array = [int(self.board[i*self.num_columns + self.num_columns//2]) for i in range(self.num_rows)]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for r in range(self.num_rows):
            row_array = [int(self.board[r*self.num_columns + c]) for c in range(self.num_columns)]
            for c in range(self.num_columns - self.num_win + 1):
                window = row_array[c:c+self.num_win]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for c in range(self.num_columns):
            col_array = [int(self.board[r*self.num_columns + c]) for r in range(self.num_rows)]
            for r in range(self.num_rows - self.num_win + 1):
                window = col_array[r:r+self.num_win]
                score += self.evaluate_window(window, piece)

        # Score positive sloped diagonal
        for r in range(self.num_rows - self.num_win + 1):
            for c in range(self.num_columns - self.num_win + 1):
                window = [self.board[(r+i)*self.num_columns + c + i] for i in range(self.num_win)]
                score += self.evaluate_window(window, piece)

        # Score negative sloped diagonal
        for r in range(self.num_win - 1, self.num_rows):
            for c in range(self.num_columns - self.num_win + 1):
                window = [self.board[(r-i)*self.num_columns + c + i] for i in range(self.num_win)]
                score += self.evaluate_window(window, piece)

        return score

    def minimax(self, depth, alpha, beta, maximizingPlayer):
        valid_locations = self.get_valid_locations()
        is_terminal = self.is_terminal_node()
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.__check_win(1):
                    return (None, 100000000000000)
                elif self.__check_win(-1):
                    return (None, -10000000000000)
                else: # Game is over, no more valid moves
                    return (None, 0)
            else: # Depth is zero
                return (None, self.score_position(1 if maximizingPlayer else -1))
        
        if maximizingPlayer:
            value = -np.inf
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(col)
                b_copy = self.board.copy()
                self.__update_board(col, 1)
                new_score = self.minimax(depth-1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                self.board = b_copy
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else: # Minimizing player
            value = np.inf
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(col)
                b_copy = self.board.copy()
                self.__update_board(col, -1)
                new_score = self.minimax(depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                self.board = b_copy
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_next_open_row(self, col):
        for r in range(self.num_rows):
            if self.board[r*self.num_columns + col] == 0:
                return r