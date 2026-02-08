import pygame
import os

from config import IMG_DIR, BGM_DIR, SOUND_DIR
from objects import Board, Pawn, Bishop, Rook, Knight, Queen, King
from typing import List, Tuple, Optional, Union


# Initialize pygame
pygame.init()

# 보드, 체스말과 기타 이미지, BGM 가져오기
img_board = pygame.image.load(os.path.join(IMG_DIR, "chess_board.png"))
img_board = pygame.transform.scale(img_board,(800,800))

img_king_b = pygame.image.load(os.path.join(IMG_DIR, "king_b.png")) # b, w = black, white
img_king_b = pygame.transform.scale(img_king_b,(100,100))
img_king_w = pygame.image.load(os.path.join(IMG_DIR, "king_w.png"))
img_king_w = pygame.transform.scale(img_king_w,(100,100))

img_queen_b = pygame.image.load(os.path.join(IMG_DIR, "queen_b.png"))
img_queen_b = pygame.transform.scale(img_queen_b,(100,100))
img_queen_b2 = pygame.transform.scale(img_queen_b,(50,50))
img_queen_w = pygame.image.load(os.path.join(IMG_DIR, "queen_w.png"))
img_queen_w = pygame.transform.scale(img_queen_w,(100,100))
img_queen_w2 = pygame.transform.scale(img_queen_w,(50,50))

img_rook_b = pygame.image.load(os.path.join(IMG_DIR, "rook_b.png"))
img_rook_b = pygame.transform.scale(img_rook_b,(100,100))
img_rook_b2 = pygame.transform.scale(img_rook_b,(50,50))
img_rook_w = pygame.image.load(os.path.join(IMG_DIR, "rook_w.png"))
img_rook_w = pygame.transform.scale(img_rook_w,(100,100))
img_rook_w2 = pygame.transform.scale(img_rook_w,(50,50))

img_bishop_b = pygame.image.load(os.path.join(IMG_DIR, "bishop_b.png"))
img_bishop_b = pygame.transform.scale(img_bishop_b,(100,100))
img_bishop_b2 = pygame.transform.scale(img_bishop_b,(50,50))
img_bishop_w = pygame.image.load(os.path.join(IMG_DIR, "bishop_w.png"))
img_bishop_w = pygame.transform.scale(img_bishop_w,(100,100))
img_bishop_w2 = pygame.transform.scale(img_bishop_w,(50,50))

img_knight_b = pygame.image.load(os.path.join(IMG_DIR, "knight_b.png"))
img_knight_b = pygame.transform.scale(img_knight_b,(100,100))
img_knight_b2 = pygame.transform.scale(img_knight_b,(50,50))
img_knight_w = pygame.image.load(os.path.join(IMG_DIR, "knight_w.png"))
img_knight_w = pygame.transform.scale(img_knight_w,(100,100))
img_knight_w2 = pygame.transform.scale(img_knight_w,(50,50))

img_pawn_b = pygame.image.load(os.path.join(IMG_DIR, "pawn_b.png"))
img_pawn_b = pygame.transform.scale(img_pawn_b,(100,100))
img_pawn_w = pygame.image.load(os.path.join(IMG_DIR, "pawn_w.png"))
img_pawn_w = pygame.transform.scale(img_pawn_w,(100,100))

# 일반적인 밝고 어두운 타일 이미지
bright_tile = pygame.image.load(os.path.join(IMG_DIR, "bright.png"))
bright_tile = pygame.transform.scale(bright_tile,(102,102))

dark_tile = pygame.image.load(os.path.join(IMG_DIR, "dark_tile.png"))
dark_tile = pygame.transform.scale(dark_tile,(102,102))

# BGM, sound
pygame.mixer.music.load(os.path.join(BGM_DIR, "Gyakuten_Kenji_2_Showdown_Suite.wav"))

sounds = {
    'place': pygame.mixer.Sound(os.path.join(SOUND_DIR, "move-self.mp3")),
    'capture': pygame.mixer.Sound(os.path.join(SOUND_DIR, "capture.mp3")),
    'promotion': pygame.mixer.Sound(os.path.join(SOUND_DIR, "승진.wav")),
    'game_end': pygame.mixer.Sound(os.path.join(SOUND_DIR, "[효과음]BOING.wav")),
    'dangerous': pygame.mixer.Sound(os.path.join(SOUND_DIR, "미스테리.wav")),
    'check': pygame.mixer.Sound(os.path.join(SOUND_DIR, "췍.mp3")),
    'blocked': pygame.mixer.Sound(os.path.join(SOUND_DIR, "디제이스탑.wav"))
}

# 좌표 기호 써있는 타일 이미지
def get_special_tiles() -> Tuple[List[pygame.Surface], List[pygame.Surface]]:
    # 좌표 기호 써있는 타일 이미지
    a1_tile = pygame.image.load(os.path.join(IMG_DIR, "a1.png"))
    a1_tile = pygame.transform.scale(a1_tile,(100,100))
    a2_tile = pygame.image.load(os.path.join(IMG_DIR, "a2.png"))
    a2_tile = pygame.transform.scale(a2_tile,(100,100))
    a3_tile = pygame.image.load(os.path.join(IMG_DIR, "a3.png"))
    a3_tile = pygame.transform.scale(a3_tile,(100,100))
    a4_tile = pygame.image.load(os.path.join(IMG_DIR, "a4.png"))
    a4_tile = pygame.transform.scale(a4_tile,(100,100))
    a5_tile = pygame.image.load(os.path.join(IMG_DIR, "a5.png"))
    a5_tile = pygame.transform.scale(a5_tile,(100,100))
    a6_tile = pygame.image.load(os.path.join(IMG_DIR, "a6.png"))
    a6_tile = pygame.transform.scale(a6_tile,(100,100))
    a7_tile = pygame.image.load(os.path.join(IMG_DIR, "a7.png"))
    a7_tile = pygame.transform.scale(a7_tile,(100,100))
    a8_tile = pygame.image.load(os.path.join(IMG_DIR, "a8.png"))
    a8_tile = pygame.transform.scale(a8_tile,(100,100))
    b1_tile = pygame.image.load(os.path.join(IMG_DIR, "b1.png"))
    b1_tile = pygame.transform.scale(b1_tile,(100,100))
    c1_tile = pygame.image.load(os.path.join(IMG_DIR, "c1.png"))
    c1_tile = pygame.transform.scale(c1_tile,(100,100))
    d1_tile = pygame.image.load(os.path.join(IMG_DIR, "d1.png"))
    d1_tile = pygame.transform.scale(d1_tile,(100,100))
    e1_tile = pygame.image.load(os.path.join(IMG_DIR, "e1.png"))
    e1_tile = pygame.transform.scale(e1_tile,(100,100))
    f1_tile = pygame.image.load(os.path.join(IMG_DIR, "f1.png"))
    f1_tile = pygame.transform.scale(f1_tile,(100,100))
    g1_tile = pygame.image.load(os.path.join(IMG_DIR, "g1.png"))
    g1_tile = pygame.transform.scale(g1_tile,(100,100))
    h1_tile = pygame.image.load(os.path.join(IMG_DIR, "h1.png"))
    h1_tile = pygame.transform.scale(h1_tile,(100,100))
    
    abc_tile = [a1_tile, b1_tile, c1_tile, d1_tile, e1_tile, f1_tile, g1_tile, h1_tile]
    num_tile = [a8_tile, a7_tile, a6_tile, a5_tile, a4_tile, a3_tile, a2_tile, a1_tile]
    
    return abc_tile, num_tile

# 선택된 기물의 가능한 이동 위치를 표시한다
def show_possible_moves(selected_piece: Union[Pawn, Bishop, Rook, Knight, Queen, King], 
                       board: Board, 
                       screen: pygame.Surface) -> None:
    for y in range(8):
        for x in range(8):
            if selected_piece.movable(board, x, y):
                # Draw a semi-transparent light gray circle
                s = pygame.Surface((80, 80), pygame.SRCALPHA)
                pygame.draw.circle(s, (200, 200, 200, 128), (40, 40), 20)
                screen.blit(s, (x*100 + 10, y*100 + 10))

# 체스판 초기화 함수
def init_chessboard(board: Board, my_color: int) -> None: # 판에 말을 세팅해놓는다
    if my_color == -1:  # Black pieces at top
        # Black pieces (top)
        Rook(board, 0, 0, 1)
        Knight(board, 1, 0, 1)
        Bishop(board, 2, 0, 1)
        Queen(board, 3, 0, 1)
        King(board, 4, 0, 1)
        Bishop(board, 5, 0, 1)
        Knight(board, 6, 0, 1)
        Rook(board, 7, 0, 1)

        for x in range(8):
            Pawn(board, x, 1, 1)

        # White pieces (bottom)
        Rook(board, 0, 7, -1)
        Knight(board, 1, 7, -1)
        Bishop(board, 2, 7, -1)
        Queen(board, 3, 7, -1)
        King(board, 4, 7, -1)
        Bishop(board, 5, 7, -1)
        Knight(board, 6, 7, -1)
        Rook(board, 7, 7, -1)

        for x in range(8):
            Pawn(board, x, 6, -1)
    else:  # White pieces at top
        # White pieces (top)
        Rook(board, 0, 0, -1)
        Knight(board, 1, 0, -1)
        Bishop(board, 2, 0, -1)
        Queen(board, 3, 0, -1)
        King(board, 4, 0, -1)
        Bishop(board, 5, 0, -1)
        Knight(board, 6, 0, -1)
        Rook(board, 7, 0, -1)

        for x in range(8):
            Pawn(board, x, 1, -1)

        # Black pieces (bottom)
        Rook(board, 0, 7, 1)
        Knight(board, 1, 7, 1)
        Bishop(board, 2, 7, 1)
        Queen(board, 3, 7, 1)
        King(board, 4, 7, 1)
        Bishop(board, 5, 7, 1)
        Knight(board, 6, 7, 1)
        Rook(board, 7, 7, 1)

        for x in range(8):
            Pawn(board, x, 6, 1)

# 초기화된 판과 세팅된 말 이미지를 띄운다
def screen_blit_initialized_board(my_color: int, screen: pygame.Surface) -> None:
    # 보드
    screen.blit(img_board,(0,0))

    if my_color == -1:
        # Black pieces (top)
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

        # White pieces (bottom)
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
    else:
        # White pieces (top)
        screen.blit(img_rook_w,(0,0))
        screen.blit(img_knight_w,(100,0))
        screen.blit(img_bishop_w,(200,0))
        screen.blit(img_queen_w,(300,0))
        screen.blit(img_king_w,(400,0))
        screen.blit(img_bishop_w,(500,0))
        screen.blit(img_knight_w,(600,0))
        screen.blit(img_rook_w,(700,0))

        for x in range(0, 800, 100):
            screen.blit(img_pawn_w,(x,100))

        # Black pieces (bottom)
        screen.blit(img_rook_b,(0,700))
        screen.blit(img_knight_b,(100,700))
        screen.blit(img_bishop_b,(200,700))
        screen.blit(img_queen_b,(300,700))
        screen.blit(img_king_b,(400,700))
        screen.blit(img_bishop_b,(500,700))
        screen.blit(img_knight_b,(600,700))
        screen.blit(img_rook_b,(700,700))

        for x in range(0, 800, 100):
            screen.blit(img_pawn_b,(x,600))

# 클릭한 위치의 좌표를 저장한다
def save_xy_selected(xy: List[int], 
                    win_xy: List[int], 
                    chessboard_xy: List[Union[str, int]], 
                    mouse_pos: Tuple[int, int]) -> None:
    # x좌표 선택
    if 0 <= mouse_pos[0] < 100:
        xy.append(0)
        win_xy.append(0)
        chessboard_xy.append("a")
    elif 100 <= mouse_pos[0] < 200:
        xy.append(1)
        win_xy.append(100)
        chessboard_xy.append("b")
    elif 200 <= mouse_pos[0] < 300:
        xy.append(2)
        win_xy.append(200)
        chessboard_xy.append("c")
    elif 300 <= mouse_pos[0] < 400:
        xy.append(3)
        win_xy.append(300)
        chessboard_xy.append("d")
    elif 400 <= mouse_pos[0] < 500:
        xy.append(4)
        win_xy.append(400)
        chessboard_xy.append("e")
    elif 500 <= mouse_pos[0] < 600:
        xy.append(5)
        win_xy.append(500)
        chessboard_xy.append("f")
    elif 600 <= mouse_pos[0] < 700:
        xy.append(6)
        win_xy.append(600)
        chessboard_xy.append("g")
    elif 700 <= mouse_pos[0] < 800:
        xy.append(7)
        win_xy.append(700)
        chessboard_xy.append("h")

    # y좌표 선택
    if 0 <= mouse_pos[1] < 100:
        xy.append(0)
        win_xy.append(0)
        chessboard_xy.append(8)
    elif 100 <= mouse_pos[1] < 200:
        xy.append(1)
        win_xy.append(100)
        chessboard_xy.append(7)
    elif 200 <= mouse_pos[1] < 300:
        xy.append(2)
        win_xy.append(200)
        chessboard_xy.append(6)
    elif 300 <= mouse_pos[1] < 400:
        xy.append(3)
        win_xy.append(300)
        chessboard_xy.append(5)
    elif 400 <= mouse_pos[1] < 500:
        xy.append(4)
        win_xy.append(400)
        chessboard_xy.append(4)
    elif 500 <= mouse_pos[1] < 600:
        xy.append(5)
        win_xy.append(500)
        chessboard_xy.append(3)
    elif 600 <= mouse_pos[1] < 700:
        xy.append(6)
        win_xy.append(600)
        chessboard_xy.append(2)
    elif 700 <= mouse_pos[1] < 800:
        xy.append(7)
        win_xy.append(700)
        chessboard_xy.append(1)

# 좌표를 PGN 형식으로 출력
def print_xy(xy: List[int], chessboard_xy: List[Union[str, int]], board: Board) -> None:
    # Convert chessboard coordinates to PGN format
    file = chessboard_xy[0]  # a-h
    rank = chessboard_xy[1]  # 1-8
    
    # Get the piece type from the board
    piece = board.board[xy[1]][xy[0]]
    piece_symbol = ""
    piece_emoji = ""
    
    if isinstance(piece, King):
        piece_symbol = "K"
        piece_emoji = "♚" if piece.color == -1 else "♔"
    elif isinstance(piece, Queen):
        piece_symbol = "Q"
        piece_emoji = "♛" if piece.color == -1 else "♕"
    elif isinstance(piece, Rook):
        piece_symbol = "R"
        piece_emoji = "♜" if piece.color == -1 else "♖"
    elif isinstance(piece, Bishop):
        piece_symbol = "B"
        piece_emoji = "♝" if piece.color == -1 else "♗"
    elif isinstance(piece, Knight):
        piece_symbol = "N"
        piece_emoji = "♞" if piece.color == -1 else "♘"
    elif isinstance(piece, Pawn):
        piece_symbol = ""
        piece_emoji = "♟" if piece.color == -1 else "♙"
    
    # Print the move in PGN format with emojis
    print(f"{piece_emoji}{piece_symbol}{file}{rank} ({xy[0]},{xy[1]})")

# 선택됬던 기물 이미지를 띄운다
def screen_blit_selected_piece(selected_piece: Union[Pawn, Bishop, Rook, Knight, Queen, King], 
                             win_xy: List[int], 
                             screen: pygame.Surface, 
                             whose_turn: int) -> None:
    if isinstance(selected_piece, King):
        if whose_turn == -1:
            screen.blit(img_king_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_king_b,(win_xy[0], win_xy[1]))
    
    elif isinstance(selected_piece, Queen):
        if whose_turn == -1:
            screen.blit(img_queen_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_queen_b,(win_xy[0], win_xy[1]))
    
    elif isinstance(selected_piece, Rook):
        if whose_turn == -1:
            screen.blit(img_rook_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_rook_b,(win_xy[0], win_xy[1]))
    
    elif isinstance(selected_piece, Knight):
        if whose_turn == -1:
            screen.blit(img_knight_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_knight_b,(win_xy[0], win_xy[1]))
    
    elif isinstance(selected_piece, Bishop):
        if whose_turn == -1:
            screen.blit(img_bishop_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_bishop_b,(win_xy[0], win_xy[1]))
    
    elif isinstance(selected_piece, Pawn):
        if whose_turn == -1:
            screen.blit(img_pawn_w,(win_xy[0], win_xy[1]))
        else:
            screen.blit(img_pawn_b,(win_xy[0], win_xy[1]))

# 기물 이미지를 업데이트한다
def screen_blit_all_pieces(board: Board, 
                         screen: pygame.Surface, 
                         whose_turn: int, 
                         game_state: str) -> None:
    # Draw red gradient circle for checkmated king if game is over
    if game_state == "Checkmate":
        for y in range(8):
            for x in range(8):
                piece = board.board[y][x]
                if isinstance(piece, King) and piece.color != whose_turn:
                    # Create a surface for the gradient circle
                    s = pygame.Surface((100, 100), pygame.SRCALPHA)
                    # Draw multiple circles with decreasing opacity for gradient effect
                    for i in range(50, 0, -1):
                        alpha = int(255 * ((50-i)/50))  # Decrease opacity as radius decreases
                        pygame.draw.circle(s, (255, 0, 0, alpha), (50, 50), i)
                    screen.blit(s, (x*100, y*100))

    for y in range(8):
        for x in range(8):
            piece = board.board[y][x]
            if isinstance(piece, King):
                if piece.color == -1:
                    screen.blit(img_king_w,(x*100, y*100))
                else:
                    screen.blit(img_king_b,(x*100, y*100))
            
            elif isinstance(piece, Queen):
                if piece.color == -1:
                    screen.blit(img_queen_w,(x*100, y*100))
                else:
                    screen.blit(img_queen_b,(x*100, y*100))
            
            elif isinstance(piece, Rook):
                if piece.color == -1:
                    screen.blit(img_rook_w,(x*100, y*100))
                else:
                    screen.blit(img_rook_b,(x*100, y*100))
            
            elif isinstance(piece, Knight):
                if piece.color == -1:
                    screen.blit(img_knight_w,(x*100, y*100))
                else:
                    screen.blit(img_knight_b,(x*100, y*100))
            
            elif isinstance(piece, Bishop):
                if piece.color == -1:
                    screen.blit(img_bishop_w,(x*100, y*100))
                else:
                    screen.blit(img_bishop_b,(x*100, y*100))
            
            elif isinstance(piece, Pawn):
                if piece.color == -1:
                    screen.blit(img_pawn_w,(x*100, y*100))
                else:
                    screen.blit(img_pawn_b,(x*100, y*100))

# 빈 타일 이미지를 띄운다
def screen_blit_empty_tile(xy: List[int], win_xy: List[int], screen: pygame.Surface) -> None:
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
def screen_blit_about_promotion(move_to_xy: List[int], 
                              move_to_win_xy: List[int], 
                              screen: pygame.Surface, 
                              whose_turn: int) -> None:
    screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)

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
def promote(whose_turn: int, 
           move_to_xy: List[int], 
           move_to_win_xy: List[int], 
           board: Board, 
           screen: pygame.Surface, 
           mouse_pos: Tuple[int, int]) -> Optional[str]:
    # 클릭한 기물로 백 폰 승진
    if whose_turn == -1:
        if move_to_win_xy[1] <= mouse_pos[1] < move_to_win_xy[1]+50: 
            if move_to_win_xy[0] <= mouse_pos[0] < move_to_win_xy[0]+50: # 퀸 클릭
                Queen(board, move_to_xy[0], move_to_xy[1], -1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)
                screen.blit(img_queen_w,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
            elif move_to_win_xy[0]+50 <= mouse_pos[0] < move_to_win_xy[0]+100: # 룩 클릭
                Rook(board, move_to_xy[0], move_to_xy[1], -1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)
                screen.blit(img_rook_w,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
        
        elif move_to_win_xy[1]+50 <= mouse_pos[1] < move_to_win_xy[1]+100: 
            if move_to_win_xy[0] <= mouse_pos[0] < move_to_win_xy[0]+50: # 비숍 클릭
                Bishop(board, move_to_xy[0], move_to_xy[1], -1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)
                screen.blit(img_bishop_w,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
            elif move_to_win_xy[0]+50 <= mouse_pos[0] < move_to_win_xy[0]+100: # 나이트 클릭
                Knight(board, move_to_xy[0], move_to_xy[1], -1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)
                screen.blit(img_knight_w,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
    
    # 클릭한 기물로 흑 폰 승진
    elif whose_turn == 1:
        if move_to_win_xy[1] <= mouse_pos[1] < move_to_win_xy[1]+50:
            if move_to_win_xy[0] <= mouse_pos[0] < move_to_win_xy[0]+50:
                Queen(board, move_to_xy[0], move_to_xy[1], 1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)
                screen.blit(img_queen_b,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
            elif move_to_win_xy[0]+50 <= mouse_pos[0] < move_to_win_xy[0]+100:
                Rook(board, move_to_xy[0], move_to_xy[1], 1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)
                screen.blit(img_rook_b,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
        
        elif move_to_win_xy[1]+50 <= mouse_pos[1] < move_to_win_xy[1]+100:
            if move_to_win_xy[0] <= mouse_pos[0] < move_to_win_xy[0]+50:
                Bishop(board, move_to_xy[0], move_to_xy[1], 1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)
                screen.blit(img_bishop_b,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
            elif move_to_win_xy[0]+50 <= mouse_pos[0] < move_to_win_xy[0]+100:
                Knight(board, move_to_xy[0], move_to_xy[1], 1)
                screen_blit_empty_tile(move_to_xy, move_to_win_xy, screen)
                screen.blit(img_knight_b,(move_to_win_xy[0],move_to_win_xy[1]))
                return "Completed"
    return None

# 승패가 결정났나 확인한다
def gamestate(board: Board, whose_turn: int) -> str:
    for y in range(8):
        for x in range(8):
            if (isinstance(board.pos(x, y), King)):
                if (((board.pos(x, y).color) == 1 and (whose_turn == -1)) or (
                    (board.pos(x, y).color) == -1 and (whose_turn == 1))):
                    return board.pos(x, y).state(board)
    return "Normal"


abc_tile, num_tile = get_special_tiles()
