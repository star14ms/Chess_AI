from main_module import *
import pygame

pygame.init()

# 보드, 체스말과 기타 이미지, BGM 가져오기
img_board = pygame.image.load("img\chess_board.png")
img_board = pygame.transform.scale(img_board,(800,800))

img_king_b = pygame.image.load("img\king_b.png") # b, w = black, white
img_king_b = pygame.transform.scale(img_king_b,(100,100))
img_king_w = pygame.image.load("img\king_w.png")
img_king_w = pygame.transform.scale(img_king_w,(100,100))

img_queen_b = pygame.image.load("img\queen_b.png")
img_queen_b = pygame.transform.scale(img_queen_b,(100,100))
img_queen_w = pygame.image.load("img\queen_w.png")
img_queen_w = pygame.transform.scale(img_queen_w,(100,100))

img_rook_b = pygame.image.load("img\\rook_b.png")
img_rook_b = pygame.transform.scale(img_rook_b,(100,100))
img_rook_w = pygame.image.load("img\\rook_w.png")
img_rook_w = pygame.transform.scale(img_rook_w,(100,100))

img_bishop_b = pygame.image.load("img\\bishop_b.png")
img_bishop_b = pygame.transform.scale(img_bishop_b,(100,100))
img_bishop_w = pygame.image.load("img\\bishop_w.png")
img_bishop_w = pygame.transform.scale(img_bishop_w,(100,100))

img_knight_b = pygame.image.load("img\knight_b.png")
img_knight_b = pygame.transform.scale(img_knight_b,(100,100))
img_knight_w = pygame.image.load("img\knight_w.png")
img_knight_w = pygame.transform.scale(img_knight_w,(100,100))

img_pawn_b = pygame.image.load("img\pawn_b.png")
img_pawn_b = pygame.transform.scale(img_pawn_b,(100,100))
img_pawn_w = pygame.image.load("img\pawn_w.png")
img_pawn_w = pygame.transform.scale(img_pawn_w,(100,100))

# 일반적인 밝고 어두운 타일 이미지
bright_tile = pygame.image.load("img\\bright.png")
bright_tile = pygame.transform.scale(bright_tile,(102,102))

dark_tile = pygame.image.load("img\dark_tile.png")
dark_tile = pygame.transform.scale(dark_tile,(102,102))

# 좌표 기호 써있는 타일 이미지
a1_tile = pygame.image.load("img\\a1.png")
a1_tile = pygame.transform.scale(a1_tile,(100,100))
a2_tile = pygame.image.load("img\\a2.png")
a2_tile = pygame.transform.scale(a2_tile,(100,100))
a3_tile = pygame.image.load("img\\a3.png")
a3_tile = pygame.transform.scale(a3_tile,(100,100))
a4_tile = pygame.image.load("img\\a4.png")
a4_tile = pygame.transform.scale(a4_tile,(100,100))
a5_tile = pygame.image.load("img\\a5.png")
a5_tile = pygame.transform.scale(a5_tile,(100,100))
a6_tile = pygame.image.load("img\\a6.png")
a6_tile = pygame.transform.scale(a6_tile,(100,100))
a7_tile = pygame.image.load("img\\a7.png")
a7_tile = pygame.transform.scale(a7_tile,(100,100))
a8_tile = pygame.image.load("img\\a8.png")
a8_tile = pygame.transform.scale(a8_tile,(100,100))
b1_tile = pygame.image.load("img\\b1.png")
b1_tile = pygame.transform.scale(b1_tile,(100,100))
c1_tile = pygame.image.load("img\\c1.png")
c1_tile = pygame.transform.scale(c1_tile,(100,100))
d1_tile = pygame.image.load("img\\d1.png")
d1_tile = pygame.transform.scale(d1_tile,(100,100))
e1_tile = pygame.image.load("img\\e1.png")
e1_tile = pygame.transform.scale(e1_tile,(100,100))
f1_tile = pygame.image.load("img\\f1.png")
f1_tile = pygame.transform.scale(f1_tile,(100,100))
g1_tile = pygame.image.load("img\\g1.png")
g1_tile = pygame.transform.scale(g1_tile,(100,100))
h1_tile = pygame.image.load("img\\h1.png")
h1_tile = pygame.transform.scale(h1_tile,(100,100))

abc_tile = [a1_tile, b1_tile, c1_tile, d1_tile, e1_tile, f1_tile, g1_tile, h1_tile]
num_tile = [a8_tile, a7_tile, a6_tile, a5_tile, a4_tile, a3_tile, a2_tile, a1_tile]

# 선택된 타일 이미지
img_selected = pygame.image.load("img\selected.png")
img_selected = pygame.transform.scale(img_selected,(100,102))

# 대전 BGM
pygame.mixer.music.load("bgm\Gyakuten_Kenji_2_Showdown_Suite.wav")

################################################################

# 체스판 초기화 함수
def do_init(): # 판에 말 세팅해놓는 함수
    
    rook_b1 = Rook(board, 0, 0, 1)
    knight_b1 = Knight(board, 0, 1, 1) #(1,0)좌표에 흑색 나이트 생성
    bishop_bb = Bishop(board, 0, 2, 1)
    queen_b = Queen(board, 0, 3, 1)
    king_b = King(board, 0, 4, 1)
    bishop_bw = Bishop(board, 0, 5, 1)
    knight_b2 = Knight(board, 0, 6, 1)
    rook_b2 = Rook(board, 0, 7, 1)

    pawn_b1 = Pawn(board, 1, 0, 1) #(0,1)좌표에 흑색 폰 생성
    pawn_b2 = Pawn(board, 1, 1, 1)
    pawn_b3 = Pawn(board, 1, 2, 1)
    pawn_b4 = Pawn(board, 1, 3, 1)
    pawn_b5 = Pawn(board, 1, 4, 1)
    pawn_b6 = Pawn(board, 1, 5, 1)
    pawn_b7 = Pawn(board, 1, 6, 1)
    pawn_b8 = Pawn(board, 1, 7, 1)

    pawn_w1 = Pawn(board, 6, 0, -1) 
    pawn_w2 = Pawn(board, 6, 1, -1)
    pawn_w3 = Pawn(board, 6, 2, -1)
    pawn_w4 = Pawn(board, 6, 3, -1)
    pawn_w5 = Pawn(board, 6, 4, -1)
    pawn_w6 = Pawn(board, 6, 5, -1)
    pawn_w7 = Pawn(board, 6, 6, -1)
    pawn_w8 = Pawn(board, 6, 7, -1)
    
    rook_w1 = Rook(board, 7, 0, -1)
    knight_w1 = Knight(board, 7, 1, -1) # (1,7) 좌표에 백색 나이트 생성
    bishop_ww = Bishop(board, 7, 2, -1)
    queen_w = Queen(board, 7, 3, -1)
    king_w = King(board, 7, 4, -1)
    bishop_wb = Bishop(board, 7, 5, -1)
    knight_w2 = Knight(board, 7, 6, -1)
    rook_w2 = Rook(board, 7, 7, -1)

def screen_blit_initialized_board(): # 판과 세팅된 말 이미지를 띄운다
    
    # 보드
    screen.blit(img_board,(0,0)) 

    # 흑의 말들
    screen.blit(img_rook_b,(0,0))
    screen.blit(img_knight_b,(100,0))
    screen.blit(img_bishop_b,(200,0))
    screen.blit(img_queen_b,(300,0))
    screen.blit(img_king_b,(400,0))
    screen.blit(img_bishop_b,(500,0))
    screen.blit(img_knight_b,(600,0))
    screen.blit(img_rook_b,(700,0))

    for x in range(0, 1000, 100):
        screen.blit(img_pawn_b,(x,100))

    # 백의 말들
    screen.blit(img_rook_w,(0,700)) 
    screen.blit(img_knight_w,(100,700))
    screen.blit(img_bishop_w,(200,700))
    screen.blit(img_queen_w,(300,700))
    screen.blit(img_king_w,(400,700))
    screen.blit(img_bishop_w,(500,700))
    screen.blit(img_knight_w,(600,700))
    screen.blit(img_rook_w,(700,700))

    for x in range(0, 1000, 100):
        screen.blit(img_pawn_w,(x,600))

# 클릭한 위치의 좌표를 저장한다
def select_piece_xy(): # 움직일 기물의 좌표를 저장

    # x좌표 선택
    if 0 <= pygame.mouse.get_pos()[0] < 100:
        selected_piece_xy.append(0)
        selected_piece_xy2.append("a")
        selected_piece_win_xy.append(0)
    elif 100 <= pygame.mouse.get_pos()[0] < 200:
        selected_piece_xy.append(1)
        selected_piece_xy2.append("b")
        selected_piece_win_xy.append(100)
    elif 200 <= pygame.mouse.get_pos()[0] < 300:
        selected_piece_xy.append(2)
        selected_piece_xy2.append("c")
        selected_piece_win_xy.append(200)
    elif 300 <= pygame.mouse.get_pos()[0] < 400:
        selected_piece_xy.append(3)
        selected_piece_xy2.append("d")
        selected_piece_win_xy.append(300)
    elif 400 <= pygame.mouse.get_pos()[0] < 500:
        selected_piece_xy.append(4)
        selected_piece_xy2.append("e")
        selected_piece_win_xy.append(400)
    elif 500 <= pygame.mouse.get_pos()[0] < 600:
        selected_piece_xy.append(5)
        selected_piece_xy2.append("f")
        selected_piece_win_xy.append(500)
    elif 600 <= pygame.mouse.get_pos()[0] < 700:
        selected_piece_xy.append(6)
        selected_piece_xy2.append("g")
        selected_piece_win_xy.append(600)
    elif 700 <= pygame.mouse.get_pos()[0] < 800:
        selected_piece_xy.append(7)
        selected_piece_xy2.append("h")
        selected_piece_win_xy.append(700)

    # y좌표 선택
    if 0 <= pygame.mouse.get_pos()[1] < 100:
        selected_piece_xy.append(0)
        selected_piece_xy2.append(8)
        selected_piece_win_xy.append(0)
    elif 100 <= pygame.mouse.get_pos()[1] < 200:
        selected_piece_xy.append(1)
        selected_piece_xy2.append(7)
        selected_piece_win_xy.append(100)
    elif 200 <= pygame.mouse.get_pos()[1] < 300:
        selected_piece_xy.append(2)
        selected_piece_xy2.append(6)
        selected_piece_win_xy.append(200)
    elif 300 <= pygame.mouse.get_pos()[1] < 400:
        selected_piece_xy.append(3)
        selected_piece_xy2.append(5)
        selected_piece_win_xy.append(300)
    elif 400 <= pygame.mouse.get_pos()[1] < 500:
        selected_piece_xy.append(4)
        selected_piece_xy2.append(4)
        selected_piece_win_xy.append(400)
    elif 500 <= pygame.mouse.get_pos()[1] < 600:
        selected_piece_xy.append(5)
        selected_piece_xy2.append(3)
        selected_piece_win_xy.append(500)
    elif 600 <= pygame.mouse.get_pos()[1] < 700:
        selected_piece_xy.append(6)
        selected_piece_xy2.append(2)
        selected_piece_win_xy.append(600)
    elif 700 <= pygame.mouse.get_pos()[1] < 800:
        selected_piece_xy.append(7)
        selected_piece_xy2.append(1)
        selected_piece_win_xy.append(700)

def select_board_xy(): # 움직일 위치의 좌표를 저장

    # x좌표 선택
    if 0 <= pygame.mouse.get_pos()[0] < 100:
        selected_board_xy.append(0)
        selected_board_xy2.append("a")
        selected_board_win_xy.append(0)
    elif 100 <= pygame.mouse.get_pos()[0] < 200:
        selected_board_xy.append(1)
        selected_board_xy2.append("b")
        selected_board_win_xy.append(100)
    elif 200 <= pygame.mouse.get_pos()[0] < 300:
        selected_board_xy.append(2)
        selected_board_xy2.append("c")
        selected_board_win_xy.append(200)
    elif 300 <= pygame.mouse.get_pos()[0] < 400:
        selected_board_xy.append(3)
        selected_board_xy2.append("d")
        selected_board_win_xy.append(300)
    elif 400 <= pygame.mouse.get_pos()[0] < 500:
        selected_board_xy.append(4)
        selected_board_xy2.append("e")
        selected_board_win_xy.append(400)
    elif 500 <= pygame.mouse.get_pos()[0] < 600:
        selected_board_xy.append(5)
        selected_board_xy2.append("f")
        selected_board_win_xy.append(500)
    elif 600 <= pygame.mouse.get_pos()[0] < 700:
        selected_board_xy.append(6)
        selected_board_xy2.append("g")
        selected_board_win_xy.append(600)
    elif 700 <= pygame.mouse.get_pos()[0] < 800:
        selected_board_xy.append(7)
        selected_board_xy2.append("h")
        selected_board_win_xy.append(700)

    # y좌표 선택
    if 0 <= pygame.mouse.get_pos()[1] < 100:
        selected_board_xy.append(0)
        selected_board_xy2.append(8)
        selected_board_win_xy.append(0)
    elif 100 <= pygame.mouse.get_pos()[1] < 200:
        selected_board_xy.append(1)
        selected_board_xy2.append(7)
        selected_board_win_xy.append(100)
    elif 200 <= pygame.mouse.get_pos()[1] < 300:
        selected_board_xy.append(2)
        selected_board_xy2.append(6)
        selected_board_win_xy.append(200)
    elif 300 <= pygame.mouse.get_pos()[1] < 400:
        selected_board_xy.append(3)
        selected_board_xy2.append(5)
        selected_board_win_xy.append(300)
    elif 400 <= pygame.mouse.get_pos()[1] < 500:
        selected_board_xy.append(4)
        selected_board_xy2.append(4)
        selected_board_win_xy.append(400)
    elif 500 <= pygame.mouse.get_pos()[1] < 600:
        selected_board_xy.append(5)
        selected_board_xy2.append(3)
        selected_board_win_xy.append(500)
    elif 600 <= pygame.mouse.get_pos()[1] < 700:
        selected_board_xy.append(6)
        selected_board_xy2.append(2)
        selected_board_win_xy.append(600)
    elif 700 <= pygame.mouse.get_pos()[1] < 800:
        selected_board_xy.append(7)
        selected_board_xy2.append(1)
        selected_board_win_xy.append(700)

# 클릭한 좌표의 정보를 출력한다
def print_selected_piece_xy():
    print("\nselecting phase")
    print("np xy :", selected_piece_xy[0], selected_piece_xy[1])
    print("chess xy :", selected_piece_xy2[0], selected_piece_xy2[1])

def print_selected_board_xy():
    print("\nmoving phase")
    print("np xy :", selected_board_xy[0], selected_board_xy[1])
    print("chess xy :", selected_board_xy2[0], selected_board_xy2[1])

# 선택된 기물 이미지를 띄운다
def screen_blit_selected_piece(win_xy):

    if str(type(selected_piece)) == "<class 'main_module.King'>":
        if whose_turn == -1:
            screen.blit(img_king_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_king_b,(win_xy[0], win_xy[1]))
    
    elif str(type(selected_piece)) == "<class 'main_module.Queen'>":
        if whose_turn == -1:
            screen.blit(img_queen_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_queen_b,(win_xy[0], win_xy[1]))
    
    elif str(type(selected_piece)) == "<class 'main_module.Rook'>":
        if whose_turn == -1:
            screen.blit(img_rook_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_rook_b,(win_xy[0], win_xy[1]))
    
    elif str(type(selected_piece)) == "<class 'main_module.Knight'>":
        if whose_turn == -1:
            screen.blit(img_knight_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_knight_b,(win_xy[0], win_xy[1]))
    
    elif str(type(selected_piece)) == "<class 'main_module.Bishop'>":
        if whose_turn == -1:
            screen.blit(img_bishop_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_bishop_b,(win_xy[0], win_xy[1]))
    
    elif str(type(selected_piece)) == "<class 'main_module.Pawn'>":
        if whose_turn == -1:
            screen.blit(img_pawn_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_pawn_b,(win_xy[0], win_xy[1]))

# 선택됬던 타일을 원상태로 되돌린다
def screen_blit_selected_tile():
    
    # 좌표 기호가 써있는 타일일 때
    if selected_piece_xy[0] == 0 or selected_piece_xy[1] == 7:
        if selected_piece_xy[1] == 7:
            screen.blit(abc_tile[selected_piece_xy[0]],(selected_piece_win_xy[0], selected_piece_win_xy[1]))
        else:
            screen.blit(num_tile[selected_piece_xy[1]],(selected_piece_win_xy[0], selected_piece_win_xy[1]))
    
    # 평범한 타일일 때
    elif selected_piece_xy[0] == 0 or selected_piece_xy[0] == 2 or selected_piece_xy[0] == 4 or selected_piece_xy[0] == 6:
        if selected_piece_xy[1] == 0 or selected_piece_xy[1] == 2 or selected_piece_xy[1] == 4 or selected_piece_xy[1] == 6:
            screen.blit(bright_tile,(selected_piece_win_xy[0], selected_piece_win_xy[1]))
        else:
            screen.blit(dark_tile,(selected_piece_win_xy[0], selected_piece_win_xy[1]))
    else:
        if selected_piece_xy[1] == 0 or selected_piece_xy[1] == 2 or selected_piece_xy[1] == 4 or selected_piece_xy[1] == 6:
            screen.blit(dark_tile,(selected_piece_win_xy[0], selected_piece_win_xy[1]))
        else:
            screen.blit(bright_tile,(selected_piece_win_xy[0], selected_piece_win_xy[1]))

################################################################

# 게임 창 띄우기
screen = pygame.display.set_mode((800,800))
pygame.display.set_caption("와! 체스!") # 창 제목
quit = False

while not quit:

    # 보드 초기화
    board = Board(-1) 
    do_init()
    
    # 보드 그래픽 초기화
    screen_blit_initialized_board() 
    pygame.display.update()

    # print(board.board) # 보드 상태 출력
    pygame.mixer.music.play(-1) # -1 : BGM 반복 재생
    whose_turn = -1 # -1: 백, 1: 흑 (백 선)

    selected_piece_xy = [] # 선택한 기물의 좌표
    selected_piece_xy2 = [] # 선택한 기물의 체스 보드에서의 좌표
    selected_piece_win_xy = [] # 선택한 기물의 화면 좌표

    selected_board_xy = [] # 움직일 좌표
    selected_board_xy2 = [] # 움직일 체스 보드에서의 좌표
    selected_board_win_xy = [] # 움직일 화면 좌표

    game_end = False

    # 게임 시작
    while not game_end:

        for event in pygame.event.get():
            
            if event.type == pygame.QUIT: # X 버튼
                game_end = True
                quit = True
            
            if event.type == pygame.MOUSEBUTTONUP: # 마우스를 눌렀다 떼는 순간
                
                # 움직일 기물 선택 단계
                if selected_piece_xy == []:
                    
                    # 클릭한 위치의 좌표 저장
                    select_piece_xy()
                    print_selected_piece_xy()
                    
                    # 클릭한 좌표의 말 가져오기
                    selected_piece = board.board[selected_piece_xy[1], selected_piece_xy[0]] # board.board는 x, y를 뒤집어 인식함
                    
                    # 자신의 말을 선택했을 때
                    if selected_piece != 0 and selected_piece.color == whose_turn:
                        print(type(selected_piece))
                        screen.blit(img_selected,(selected_piece_win_xy[0], selected_piece_win_xy[1]))
                        screen_blit_selected_piece(selected_piece_win_xy)

                    else: # 빈공간이나 상대의 말을 선택했을 때 다시 선택
                        print("Try again")
                        selected_piece_xy = []
                        selected_piece_xy2 = []
                        selected_piece_win_xy = []

                # 움직일 위치 선택 단계
                else:

                    # 클릭한 위치의 좌표 저장
                    select_board_xy()
                    screen_blit_selected_tile()
                    
                    # move 함수 발동!
                    # pawn1.move(board, 1)
                    # pawn2.move(board, 1)

                    # 갈 수 있는 곳을 선택할 경우
                    if selected_piece_xy != selected_board_xy: # and move() != False

                        print_selected_board_xy()
                        screen_blit_selected_piece(selected_board_win_xy)

                    # 갈 수 없는 곳이나, 선택 했던 곳을 또 선택할 경우 선택 취소
                    else:
                        screen_blit_selected_piece(selected_piece_win_xy)
                        print("\nUnselect")

                    selected_piece_xy = []
                    selected_piece_xy2 = []
                    selected_piece_win_xy = []
                    selected_board_xy = []
                    selected_board_xy2 = []
                    selected_board_win_xy = []

                pygame.display.update()

pygame.quit()