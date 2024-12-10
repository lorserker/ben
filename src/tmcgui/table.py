from tmcgui.board import Board


class Table:
    """
    Class for handling empty or full tables.
    """

    def __init__(self, table_id):
        self.id = table_id
        self.players = [None, None, None, None]  # S, W, N, E
        self.board = None
        self.empty = True
        self.board_history = []

    def is_full(self):
        """
        Checking that the table is full or empty.
        :return: boolean
        """
        if all(p for p in self.connected):
            self.empty = False
            return True
        return False

    def set_player(self, index, username):
        """
        Setting player on the specific table seat.
        :param index: int
        :param username: string
        :return: None
        """
        self.players[index] = (username)

    def remove_player(self, index):
        """
        Removing player from the specific table seat.
        :param index: int
        :return: None
        """
        self.players[index] = None

    def __repr__(self):
        return f"Table nr {self.id}"

    def next_board(self, board_id):
        """
        Initializing new board with specific ID.
        :return: None
        """
        self.board = Board(board_id)
        self.board.players = self.players
