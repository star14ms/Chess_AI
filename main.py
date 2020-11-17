from main_module import *
import pygame

board = Board(False)

def do_init(): #판에 말 세팅해놓는 함수
    pass

pawn1 = Pawn(board, 0, 0, False) #(0,0)좌표에 백색 폰 생성
pawn2 = Pawn(board, 1, 1, True) #(0,0)좌표에 흑색 폰 생성
pawn1.move(board, 1)
pawn2.move(board, 1)



pygame.init()

def init_board_img(): # 보드를 초기화한 상태의 이미지를 띄우는 함수
    
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

# 게임 창
screen = pygame.display.set_mode((800,800))
pygame.display.set_caption("와! 체스!")
quit = False

while not quit:
    
    init_board_img()
    pygame.display.update()
    
    for event in pygame.event.get():
        
        if event.type == pygame.QUIT:
            quit = True
        
        if event.type == pygame.MOUSEBUTTONUP:
            pygame.display.update()

pygame.quit()