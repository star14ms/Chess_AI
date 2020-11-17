from main_module import *
import pygame

pygame.init()

# 보드, 체스말 이미지 가져오기
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

img_rook_b = pygame.image.load("img\Rook_b.png")
img_rook_b = pygame.transform.scale(img_rook_b,(100,100))
img_rook_w = pygame.image.load("img\Rook_w.png")
img_rook_w = pygame.transform.scale(img_rook_w,(100,100))

img_bishop_b = pygame.image.load("img\Bishop_b.png")
img_bishop_b = pygame.transform.scale(img_bishop_b,(100,100))
img_bishop_w = pygame.image.load("img\Bishop_w.png")
img_bishop_w = pygame.transform.scale(img_bishop_w,(100,100))

img_knight_b = pygame.image.load("img\knight_b.png")
img_knight_b = pygame.transform.scale(img_knight_b,(100,100))
img_knight_w = pygame.image.load("img\knight_w.png")
img_knight_w = pygame.transform.scale(img_knight_w,(100,100))

img_pawn_b = pygame.image.load("img\pawn_b.png")
img_pawn_b = pygame.transform.scale(img_pawn_b,(100,100))
img_pawn_w = pygame.image.load("img\pawn_w.png")
img_pawn_w = pygame.transform.scale(img_pawn_w,(100,100))

# 게임 창 띄우기
screen = pygame.display.set_mode((800,800))
pygame.display.set_caption("와! 체스!") # 창 제목
quit = False

# 보드 초기화 함수
def do_init(): # 판에 말 세팅해놓는 함수
    
    rook_w1 = Rook(board, 0, 0, False)
    knight_w1 = Knight(board, 0, 1, False) # (1,0) 좌표에 백색 나이트 생성
    bishop_w1 = Bishop(board, 0, 2, False)
    king_w = King(board, 0, 3, False)
    queen_w = Queen(board, 0, 4, False)
    bishop_w2 = Bishop(board, 0, 5, False)
    knight_w2 = Knight(board, 0, 6, False)
    rook_w2 = Rook(board, 0, 7, False)

    pawn_w1 = Pawn(board, 1, 0, False) #(0,1)좌표에 백색 폰 생성
    pawn_w2 = Pawn(board, 1, 1, False)
    pawn_w3 = Pawn(board, 1, 2, False)
    pawn_w4 = Pawn(board, 1, 3, False)
    pawn_w5 = Pawn(board, 1, 4, False)
    pawn_w6 = Pawn(board, 1, 5, False)
    pawn_w7 = Pawn(board, 1, 6, False)
    pawn_w8 = Pawn(board, 1, 7, False)

    pawn_b1 = Pawn(board, 6, 0, True) #(0,6)좌표에 흑색 폰 생성
    pawn_b2 = Pawn(board, 6, 1, True)
    pawn_b3 = Pawn(board, 6, 2, True)
    pawn_b4 = Pawn(board, 6, 3, True)
    pawn_b5 = Pawn(board, 6, 4, True)
    pawn_b6 = Pawn(board, 6, 5, True)
    pawn_b7 = Pawn(board, 6, 6, True)
    pawn_b8 = Pawn(board, 6, 7, True)

    rook_b1 = Rook(board, 7, 0, True)
    knight_b1 = Knight(board, 7, 1, True)
    bishop_b1 = Bishop(board, 7, 2, True)
    king_b = King(board, 7, 3, True)
    queen_b = Queen(board, 7, 4, True)
    bishop_b2 = Bishop(board, 7, 5, True)
    knight_b2 = Knight(board, 7, 6, True)
    rook_b2 = Rook(board, 7, 7, True)

def screen_blit_initialized_board(): # 판과 세팅된 말 이미지를 띄우는 함수
    
    # 보드
    screen.blit(img_board,(0,0)) 

    # 흑의 말들
    screen.blit(img_rook_w,(0,0)) 
    screen.blit(img_knight_w,(100,0))
    screen.blit(img_bishop_w,(200,0))
    screen.blit(img_king_w,(300,0))
    screen.blit(img_queen_w,(400,0))
    screen.blit(img_bishop_w,(500,0))
    screen.blit(img_knight_w,(600,0))
    screen.blit(img_rook_w,(700,0))

    for x in range(0, 1000, 100):
        screen.blit(img_pawn_w,(x,100))

    # 백의 말들
    screen.blit(img_rook_b,(0,700)) 
    screen.blit(img_knight_b,(100,700))
    screen.blit(img_bishop_b,(200,700))
    screen.blit(img_king_b,(300,700))
    screen.blit(img_queen_b,(400,700))
    screen.blit(img_bishop_b,(500,700))
    screen.blit(img_knight_b,(600,700))
    screen.blit(img_rook_b,(700,700))

    for x in range(0, 1000, 100):
        screen.blit(img_pawn_b,(x,600))

while not quit:

    # 게임 초기화
    board = Board(False) # 보드 초기화
    do_init()

    screen_blit_initialized_board() # 보드 그래픽 초기화
    pygame.display.update()

    # print(board.board) # 보드 상태 출력
    game_end = False

    # 게임 시작
    while not game_end:

        for event in pygame.event.get():
            
            if event.type == pygame.QUIT: # X 버튼
                game_end = True
                quit = True
            
            if event.type == pygame.MOUSEBUTTONUP: # 마우스를 눌렀다 떼는 순간

                # pawn1.move(board, 1)
                # pawn2.move(board, 1)

                pygame.display.update()

pygame.quit()