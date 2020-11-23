from main_module import *

board = Board(-1)

bishop1 = Bishop(board, 3, 3, -1)
bishop2 = Bishop(board, 4, 4, 1)
bishop3 = Bishop(board, 5, 5, 1)
print(bishop1.move(board, 5, 5))
print(bishop1.move(board, 4, 4))
print(bishop1.move(board, 5, 5))