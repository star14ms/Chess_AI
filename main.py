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
img_queen_b2 = pygame.transform.scale(img_queen_b,(50,50))
img_queen_w = pygame.image.load("img\queen_w.png")
img_queen_w = pygame.transform.scale(img_queen_w,(100,100))
img_queen_w2 = pygame.transform.scale(img_queen_w,(50,50))

img_rook_b = pygame.image.load("img\\rook_b.png")
img_rook_b = pygame.transform.scale(img_rook_b,(100,100))
img_rook_b2 = pygame.transform.scale(img_rook_b,(50,50))
img_rook_w = pygame.image.load("img\\rook_w.png")
img_rook_w = pygame.transform.scale(img_rook_w,(100,100))
img_rook_w2 = pygame.transform.scale(img_rook_w,(50,50))

img_bishop_b = pygame.image.load("img\\bishop_b.png")
img_bishop_b = pygame.transform.scale(img_bishop_b,(100,100))
img_bishop_b2 = pygame.transform.scale(img_bishop_b,(50,50))
img_bishop_w = pygame.image.load("img\\bishop_w.png")
img_bishop_w = pygame.transform.scale(img_bishop_w,(100,100))
img_bishop_w2 = pygame.transform.scale(img_bishop_w,(50,50))

img_knight_b = pygame.image.load("img\knight_b.png")
img_knight_b = pygame.transform.scale(img_knight_b,(100,100))
img_knight_b2 = pygame.transform.scale(img_knight_b,(50,50))
img_knight_w = pygame.image.load("img\knight_w.png")
img_knight_w = pygame.transform.scale(img_knight_w,(100,100))
img_knight_w2 = pygame.transform.scale(img_knight_w,(50,50))

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
img_selected = pygame.transform.scale(img_selected,(102,102))

# BGM, sound
pygame.mixer.music.load("bgm\Gyakuten_Kenji_2_Showdown_Suite.wav")
sound_place = pygame.mixer.Sound("sound\체스말_놓기.wav")
sound_promotion = pygame.mixer.Sound("sound\승진.wav")
sound_game_end = pygame.mixer.Sound("sound\[효과음]BOING.wav")
sound_check = pygame.mixer.Sound("sound\미스테리.wav")
sound_dangerous = pygame.mixer.Sound("sound\디제이스탑.wav")

################################################################

# 체스판 초기화 함수
def do_init(): # 판에 말을 세팅해놓는다
    
    rook_b1 = Rook(board, 0, 0, 1)
    knight_b1 = Knight(board, 1, 0, 1) #(1,0)좌표에 흑색 나이트 생성
    bishop_bb = Bishop(board, 2, 0, 1)
    queen_b = Queen(board, 3, 0, 1)
    king_b = King(board, 4, 0, 1)
    bishop_bw = Bishop(board, 5, 0, 1)
    knight_b2 = Knight(board, 6, 0, 1)
    rook_b2 = Rook(board, 7, 0, 1)

    pawn_b1 = Pawn(board, 0, 1, 1) #(0,1)좌표에 흑색 폰 생성
    pawn_b2 = Pawn(board, 1, 1, 1)
    pawn_b3 = Pawn(board, 2, 1, 1)
    pawn_b4 = Pawn(board, 3, 1, 1)
    pawn_b5 = Pawn(board, 4, 1, 1)
    pawn_b6 = Pawn(board, 5, 1, 1)
    pawn_b7 = Pawn(board, 6, 1, 1)
    pawn_b8 = Pawn(board, 7, 1, 1)

    pawn_w1 = Pawn(board, 0, 6, -1) 
    pawn_w2 = Pawn(board, 1, 6, -1)
    pawn_w3 = Pawn(board, 2, 6, -1)
    pawn_w4 = Pawn(board, 3, 6, -1)
    pawn_w5 = Pawn(board, 4, 6, -1)
    pawn_w6 = Pawn(board, 5, 6, -1)
    pawn_w7 = Pawn(board, 6, 6, -1)
    pawn_w8 = Pawn(board, 7, 6, -1)
    
    rook_w1 = Rook(board, 0, 7, -1)
    knight_w1 = Knight(board, 1, 7, -1) # (1,7) 좌표에 백색 나이트 생성
    bishop_ww = Bishop(board, 2, 7, -1)
    queen_w = Queen(board, 3, 7, -1)
    king_w = King(board, 4, 7, -1)
    bishop_wb = Bishop(board, 5, 7, -1)
    knight_w2 = Knight(board, 6, 7, -1)
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

    for x in range(0, 800, 100):
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

    for x in range(0, 800, 100):
        screen.blit(img_pawn_w,(x,600))

# 클릭한 위치의 좌표를 저장한다
def save_xy_selected(xy, win_xy, chessboard_xy): # 선택한 기물의 좌표를 저장
    
    # x좌표 선택
    if 0 <= pygame.mouse.get_pos()[0] < 100:
        xy.append(0)
        win_xy.append(0)
        chessboard_xy.append("a")
    elif 100 <= pygame.mouse.get_pos()[0] < 200:
        xy.append(1)
        win_xy.append(100)
        chessboard_xy.append("b")
    elif 200 <= pygame.mouse.get_pos()[0] < 300:
        xy.append(2)
        win_xy.append(200)
        chessboard_xy.append("c")
    elif 300 <= pygame.mouse.get_pos()[0] < 400:
        xy.append(3)
        win_xy.append(300)
        chessboard_xy.append("d")
    elif 400 <= pygame.mouse.get_pos()[0] < 500:
        xy.append(4)
        win_xy.append(400)
        chessboard_xy.append("e")
    elif 500 <= pygame.mouse.get_pos()[0] < 600:
        xy.append(5)
        win_xy.append(500)
        chessboard_xy.append("f")
    elif 600 <= pygame.mouse.get_pos()[0] < 700:
        xy.append(6)
        win_xy.append(600)
        chessboard_xy.append("g")
    elif 700 <= pygame.mouse.get_pos()[0] < 800:
        xy.append(7)
        win_xy.append(700)
        chessboard_xy.append("h")

    # y좌표 선택
    if 0 <= pygame.mouse.get_pos()[1] < 100:
        xy.append(0)
        win_xy.append(0)
        chessboard_xy.append(8)
    elif 100 <= pygame.mouse.get_pos()[1] < 200:
        xy.append(1)
        win_xy.append(100)
        chessboard_xy.append(7)
    elif 200 <= pygame.mouse.get_pos()[1] < 300:
        xy.append(2)
        win_xy.append(200)
        chessboard_xy.append(6)
    elif 300 <= pygame.mouse.get_pos()[1] < 400:
        xy.append(3)
        win_xy.append(300)
        chessboard_xy.append(5)
    elif 400 <= pygame.mouse.get_pos()[1] < 500:
        xy.append(4)
        win_xy.append(400)
        chessboard_xy.append(4)
    elif 500 <= pygame.mouse.get_pos()[1] < 600:
        xy.append(5)
        win_xy.append(500)
        chessboard_xy.append(3)
    elif 600 <= pygame.mouse.get_pos()[1] < 700:
        xy.append(6)
        win_xy.append(600)
        chessboard_xy.append(2)
    elif 700 <= pygame.mouse.get_pos()[1] < 800:
        xy.append(7)
        win_xy.append(700)
        chessboard_xy.append(1)

# 클릭한 좌표의 정보를 출력한다
def print_xy_selected(move_from_xy, move_from_chessboard_xy):
    print("selecting phase")
    print("chess xy :", move_from_chessboard_xy[0], move_from_chessboard_xy[1])
    print("np xy :", move_from_xy[0], move_from_xy[1])

def print_xy_move_to(move_to_xy, move_to_chessboard_xy):
    print("moving phase")
    print("chess xy :", move_to_chessboard_xy[0], move_to_chessboard_xy[1])
    print("np xy :", move_to_xy[0], move_to_xy[1])

# 선택됬던 기물 이미지를 띄운다
def screen_blit_selected_piece(selected_piece, win_xy):

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

# 기물 이미지를 업데이트한다
def screen_blit_all_pieces(board):

    for y in range(8):
        for x in range(8):

            if str(type(board.board[y][x])) == "<class 'main_module.King'>":
                if board.board[y][x].color == -1:
                    screen.blit(img_king_w,(x*100, y*100))
                else:
                    screen.blit(img_king_b,(x*100, y*100))
            
            elif str(type(board.board[y][x])) == "<class 'main_module.Queen'>":
                if board.board[y][x].color == -1:
                    screen.blit(img_queen_w,(x*100, y*100))
                else:
                    screen.blit(img_queen_b,(x*100, y*100))
            
            elif str(type(board.board[y][x])) == "<class 'main_module.Rook'>":
                if board.board[y][x].color == -1:
                    screen.blit(img_rook_w,(x*100, y*100))
                else:
                    screen.blit(img_rook_b,(x*100, y*100))
            
            elif str(type(board.board[y][x])) == "<class 'main_module.Knight'>":
                if board.board[y][x].color == -1:
                    screen.blit(img_knight_w,(x*100, y*100))
                else:
                    screen.blit(img_knight_b,(x*100, y*100))
            
            elif str(type(board.board[y][x])) == "<class 'main_module.Bishop'>":
                if board.board[y][x].color == -1:
                    screen.blit(img_bishop_w,(x*100, y*100))
                else:
                    screen.blit(img_bishop_b,(x*100, y*100))
            
            elif str(type(board.board[y][x])) == "<class 'main_module.Pawn'>":
                if board.board[y][x].color == -1:
                    screen.blit(img_pawn_w,(x*100, y*100))
                else:
                    screen.blit(img_pawn_b,(x*100, y*100))

# 빈 타일 이미지를 띄운다
def screen_blit_empty_tile(xy, win_xy):
    
    # 좌표 기호가 써있는 타일일 때
    if xy[0] == 0 or xy[1] == 7:
        if xy[1] == 7:
            screen.blit(abc_tile[xy[0]],(win_xy[0], win_xy[1]))
        else:
            screen.blit(num_tile[xy[1]],(win_xy[0], win_xy[1]))
    
    # 평범한 타일일 때
    elif xy[0] == 0 or xy[0] == 2 or xy[0] == 4 or xy[0] == 6:
        if xy[1] == 0 or xy[1] == 2 or xy[1] == 4 or xy[1] == 6:
            screen.blit(bright_tile,(win_xy[0], win_xy[1]))
        else:
            screen.blit(dark_tile,(win_xy[0], win_xy[1]))
    else:
        if xy[1] == 0 or xy[1] == 2 or xy[1] == 4 or xy[1] == 6:
            screen.blit(dark_tile,(win_xy[0], win_xy[1]))
        else:
            screen.blit(bright_tile,(win_xy[0], win_xy[1]))

# 승진할 기물을 선택하라는 이미지를 띄운다
def screen_blit_about_promotion(move_to_xy, move_to_win_xy):

    screen_blit_empty_tile(move_to_xy, move_to_win_xy)

    if whose_turn == -1:
        screen.blit(img_queen_w2,(move_to_win_xy[0],move_to_win_xy[1]))
        screen.blit(img_rook_w2,(move_to_win_xy[0]+50,move_to_win_xy[1]))
        screen.blit(img_bishop_w2,(move_to_win_xy[0],move_to_win_xy[1]+50))
        screen.blit(img_knight_w2,(move_to_win_xy[0]+50,move_to_win_xy[1]+50))
    else:
        screen.blit(img_queen_b2,(move_to_win_xy[0],move_to_win_xy[1]))
        screen.blit(img_rook_b2,(move_to_win_xy[0]+50,move_to_win_xy[1]))
        screen.blit(img_bishop_b2,(move_to_win_xy[0],move_to_win_xy[1]+50))
        screen.blit(img_knight_b2,(move_to_win_xy[0]+50,move_to_win_xy[1]+50))

# 승진할 기물을 클릭하면 승진시킨다
def promote(whose_turn, move_to_xy, move_to_win_xy):

    # 클릭한 기물로 백 폰 승진
    if whose_turn == -1:

        if move_to_win_xy[1] <= pygame.mouse.get_pos()[1] < move_to_win_xy[1]+50: 
            if move_to_win_xy[0] <= pygame.mouse.get_pos()[0] < move_to_win_xy[0]+50: # 퀸 클릭
                Queen(board, move_to_xy[0], move_to_xy[1], -1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy)
                screen.blit(img_queen_w,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
            elif move_to_win_xy[0]+50 <= pygame.mouse.get_pos()[0] < move_to_win_xy[0]+100: # 룩 클릭
                Rook(board, move_to_xy[0], move_to_xy[1], -1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy)
                screen.blit(img_rook_w,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
        
        elif move_to_win_xy[1]+50 <= pygame.mouse.get_pos()[1] < move_to_win_xy[1]+100: 
            if move_to_win_xy[0] <= pygame.mouse.get_pos()[0] < move_to_win_xy[0]+50: # 비숍 클릭
                Bishop(board, move_to_xy[0], move_to_xy[1], -1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy)
                screen.blit(img_bishop_w,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
            elif move_to_win_xy[0]+50 <= pygame.mouse.get_pos()[0] < move_to_win_xy[0]+100: # 나이트 클릭
                Knight(board, move_to_xy[0], move_to_xy[1], -1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy)
                screen.blit(img_knight_w,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
    
    # 클릭한 기물로 흑 폰 승진
    elif whose_turn == 1:
        
        if move_to_win_xy[1] <= pygame.mouse.get_pos()[1] < move_to_win_xy[1]+50:
            if move_to_win_xy[0] <= pygame.mouse.get_pos()[0] < move_to_win_xy[0]+50:
                Queen(board, move_to_xy[0], move_to_xy[1], 1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy)
                screen.blit(img_queen_b,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
            elif move_to_win_xy[0]+50 <= pygame.mouse.get_pos()[0] < move_to_win_xy[0]+100:
                Rook(board, move_to_xy[0], move_to_xy[1], 1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy)
                screen.blit(img_rook_b,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
        
        elif move_to_win_xy[1]+50 <= pygame.mouse.get_pos()[1] < move_to_win_xy[1]+100:
            if move_to_win_xy[0] <= pygame.mouse.get_pos()[0] < move_to_win_xy[0]+50:
                Bishop(board, move_to_xy[0], move_to_xy[1], 1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy)
                screen.blit(img_bishop_b,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
            elif move_to_win_xy[0]+50 <= pygame.mouse.get_pos()[0] < move_to_win_xy[0]+100:
                Knight(board, move_to_xy[0], move_to_xy[1], 1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy)
                screen.blit(img_knight_b,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"

# 승패가 결정났나 확인한다
def gamestate(board):
    for y in range(8):
        for x in range(8):
            if (type(board.pos(x, y)) == King):
                if (((board.pos(x, y).color) == 1 and (whose_turn == -1)) or (
                    (board.pos(x, y).color) == -1 and (whose_turn == 1))):
                    return board.pos(x, y).checkmate(board)
    return False

################################################################

# 게임 창 띄우기
screen = pygame.display.set_mode((800,800))
pygame.display.set_caption("와! 체스!") # 창 제목
quit = False

while not quit:
    
    # 보드 초기화
    board = Board(-1) # -1 : 보드 앞쪽이 흰색 진영
    do_init()
    
    # 보드 그래픽 초기화
    screen_blit_initialized_board()
    pygame.display.update()
    
    # print(board.board) # 보드 상태 출력
    pygame.mixer.music.play(-1) # -1 : BGM 반복 재생
    whose_turn = -1 # -1: 백, 1: 흑 (백 선)
    
    move_from_xy = [] # 움직일 대상 좌표
    move_from_chessboard_xy = []
    move_from_win_xy = []

    move_to_xy = [] # 움직일 위치 좌표
    move_to_chessboard_xy = []
    move_to_win_xy = []
    
    promotionable = False
    game_end = False
    game_over = False
    print("-" * 64)

    # 게임 시작
    while not game_end:
    
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT: # X 버튼
                game_end = True
                quit = True
            
            elif event.type == pygame.KEYDOWN:
                
                if event.key == pygame.K_F5: # F5 버튼
                    print("Reset")
                    game_end = True
                
                elif event.key == pygame.K_RETURN and game_over:
                    game_end = True
            
            elif event.type == pygame.MOUSEBUTTONUP and not game_over: # 마우스를 눌렀다 떼는 순간
                
                if not promotionable:
                    selected_xy = [] # 선택한 좌표
                    selected_win_xy = [] 
                    selected_chessboard_xy = []
                    save_xy_selected(selected_xy, selected_win_xy, selected_chessboard_xy)
                    selected_xy_piece = board.board[selected_xy[1]][selected_xy[0]] # board.board는 x, y를 뒤집어 인식함

                # 폰 승진 단계
                if promotionable:

                    # 승진할 기물을 클릭하면 승진시키고 턴 전환
                    if promote(whose_turn, move_to_xy, move_to_win_xy) == "Completed":
                        
                        pygame.mixer.Sound.play(sound_promotion)
                        promotionable = False
                        game_state = gamestate(board)
                        whose_turn *= -1

                        if game_state == "Check":
                            pygame.mixer.Sound.play(sound_check)
                        elif game_state == "Checkmate" and game_state == "Stalemate":
                            game_over = True
                            pygame.mixer.Sound.play(sound_game_end)
                            pygame.mixer.music.stop()
                        
                        move_from_xy = []
                        move_from_win_xy = []
                        move_from_chessboard_xy = []
                        move_to_xy = []
                        move_to_win_xy = []
                        move_to_chessboard_xy = []
                
                # 자신의 말을 선택했을 때
                elif (selected_xy_piece.color == whose_turn):
                    
                    # 이미 말을 선택했었다면 그 말 선택 취소
                    if move_from_xy != []:
                        screen_blit_empty_tile(move_from_xy, move_from_win_xy)
                        screen_blit_selected_piece(board.pos(move_from_xy[0], move_from_xy[1]), move_from_win_xy)
                        move_from_xy = []
                        move_from_win_xy = []
                        move_from_chessboard_xy = []

                    # 클릭한 위치의 좌표와 말 저장
                    move_from_xy, move_from_win_xy, move_from_chessboard_xy = selected_xy, selected_win_xy, selected_chessboard_xy
                    selected_piece = board.board[move_from_xy[1]][move_from_xy[0]] # board.board는 x, y를 뒤집어 인식함
                    
                    # 정보 출력하고, 선택된 기물 그래픽으로 표시
                    print_xy_selected(move_from_xy, move_from_chessboard_xy)
                    print(type(selected_piece), "\n")
                    screen.blit(img_selected,(move_from_win_xy[0], move_from_win_xy[1]))
                    screen_blit_selected_piece(selected_piece, move_from_win_xy)
                    
                # 자신의 말이 선택되어 있을 때
                elif move_from_xy != []:
                    
                    # 클릭한 위치의 좌표 저장
                    move_to_xy, move_to_win_xy, move_to_chessboard_xy = selected_xy, selected_win_xy, selected_chessboard_xy
                    
                    # 갈 수 있는 곳을 선택하여 말이 움직인 경우
                    if selected_piece.move(board, move_to_xy[0], move_to_xy[1]):
                        
                        pygame.mixer.Sound.play(sound_place)
                        print_xy_move_to(move_to_xy, move_to_chessboard_xy)

                        # 폰이 승진할 수 있다면 
                        if (type(selected_piece) == Pawn) and (
                            (((whose_turn == -1) and (move_to_xy[1] == 0)) or (
                            (whose_turn == 1) and (move_to_xy[1] == 7)))):

                            # 승진할 기물을 선택하라는 이미지 띄우기
                            screen_blit_empty_tile(move_from_xy, move_from_win_xy)
                            screen_blit_about_promotion(move_to_xy, move_to_win_xy)

                            # 폰 승진 단계로 이동
                            promotionable = True

                        # 승진 할 수 없다면 턴 종료
                        else:
                            screen.blit(img_board,(0,0))
                            screen_blit_all_pieces(board)
                            game_state = gamestate(board)
                            whose_turn *= -1
                            
                            # 체크, 체크메이트, 스테일메이트 여부 확인
                            if game_state == "Check":
                                pygame.mixer.Sound.play(sound_check)
                                print("\nCheck!")

                            elif game_state == "Checkmate" or game_state == "Stalemate":
                                pygame.mixer.Sound.play(sound_game_end)
                                pygame.mixer.music.stop()
                                game_over = True
                                if game_state == "Checkmate":
                                    print("\nCheckmate!")
                                else:
                                    print("\nStalemate!")

                        print("-" * 64)

                    # 갈 수 없는 곳이나, 선택 했던 곳을 또 선택할 경우 선택 취소
                    else:
                        if selected_piece.moveable(board, move_to_xy[0], move_to_xy[1]):
                            pygame.mixer.Sound.play(sound_dangerous)
                        screen_blit_empty_tile(move_from_xy, move_from_win_xy)
                        screen_blit_selected_piece(selected_piece, move_from_win_xy)
                        print("Try again\n")
                    
                    # 턴 전환 혹은 선택 취소
                    if not promotionable:
                        move_from_xy = []
                        move_from_win_xy = []
                        move_from_chessboard_xy = []
                        move_to_xy = []
                        move_to_win_xy = []
                        move_to_chessboard_xy = []
                
                # 자신의 말을 선택하지 않았을 때
                else: 
                    print("Try again\n")
                    move_from_xy = []
                    move_from_win_xy = []
                    move_from_chessboard_xy = []

                # print(board.board)
                pygame.display.update()

pygame.quit()