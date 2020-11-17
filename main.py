from main_module import *

board = Board(False)

def do_init(): #판에 말 세팅해놓는 함수
    pass

pawn1 = Pawn(board, 0, 0, False) #(0,0)좌표에 백색 폰 생성
pawn2 = Pawn(board, 1, 1, True) #(0,0)좌표에 흑색 폰 생성
pawn1.move(board, 1)
pawn2.move(board, 1)