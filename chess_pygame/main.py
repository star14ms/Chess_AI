import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT
from objects import Board, Pawn
from modules import (
    init_chessboard,
    screen_blit_initialized_board,
    show_possible_moves,
    promote,
    gamestate,
    screen_blit_empty_tile,
    screen_blit_selected_piece,
    screen_blit_all_pieces,
    screen_blit_about_promotion,
    print_xy,
    save_xy_selected,
    img_board,
    sounds,
)


def initialize_game():
    # 게임 변수 초기화
    my_color = -1 # 1: black, -1: white
    board = Board(my_color) # -1 : 보드 앞쪽이 흰색 진영
    init_chessboard(board, my_color)

    # 보드 그래픽 초기화
    screen_blit_initialized_board(my_color, screen)
    pygame.display.update()
    
    pygame.mixer.music.play(-1) # -1 : BGM 반복 재생
    whose_turn = -1 # -1: 백, 1: 흑 (백 선)
    
    move_from_xy = [] # 움직일 대상 좌표
    move_from_win_xy = []

    move_to_xy = [] # 움직일 위치 좌표
    move_to_win_xy = []
    move_to_chessboard_xy = []
    
    game_state = "Normal"
    promotionable = False
    game_end = False
    game_over = False
    print("-" * 64)

    return board, my_color, whose_turn, move_from_xy, move_from_win_xy, \
        move_to_xy, move_to_win_xy, move_to_chessboard_xy, \
        game_state, promotionable, game_end, game_over


def run_game():
    global is_muted

    board, my_color, whose_turn, move_from_xy, move_from_win_xy, \
        move_to_xy, move_to_win_xy, move_to_chessboard_xy, \
        game_state, promotionable, game_end, game_over = initialize_game()

    while not game_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # X 버튼
                game_end = True
                return True  # Return True to quit the game
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F5: # F5 버튼
                    print("Reset")
                    game_end = True
                    return False  # Return False to restart the game
                
                elif (event.key == pygame.K_SPACE or event.key == pygame.K_RETURN) and game_over:
                    game_end = True
                    return False  # Return False to restart the game

                elif event.key == pygame.K_ESCAPE: # ESC 버튼
                    game_end = True
                    return True  # Return True to quit the game
                
                elif event.key == pygame.K_m: # M 버튼
                    is_muted = not is_muted
                    if is_muted:
                        pygame.mixer.music.pause()
                    else:
                        pygame.mixer.music.unpause()
            
            elif event.type == pygame.MOUSEBUTTONUP and not game_over: # 마우스를 눌렀다 떼는 순간
                if not promotionable:
                    selected_xy = [] # 선택한 좌표
                    selected_win_xy = [] 
                    selected_chessboard_xy = []
                    save_xy_selected(selected_xy, selected_win_xy, selected_chessboard_xy, pygame.mouse.get_pos())
                    selected_xy_piece = board.board[selected_xy[1]][selected_xy[0]] # board.board는 x, y를 뒤집어 인식함

                # 폰 승진 단계
                if promotionable:
                    # 승진할 기물을 클릭하면 승진시키고 턴 전환
                    if promote(whose_turn, move_to_xy, move_to_win_xy, board, screen, pygame.mouse.get_pos()) == "Completed":
                        if not is_muted:
                            sounds['promotion'].play()
                        promotionable = False
                        game_state = gamestate(board, whose_turn)
                        whose_turn *= -1

                        if game_state == "Check":
                            if not is_muted:
                                sounds['check'].play()
                            print("Check!")
                        elif game_state == "Checkmate" or game_state == "Stalemate":
                            game_over = True
                            if not is_muted:
                                sounds['game_end'].play()
                                pygame.mixer.music.stop()

                        move_from_xy = []
                        move_from_win_xy = []
                        move_to_xy = []
                        move_to_win_xy = []
                        move_to_chessboard_xy = []
                
                # 자신의 말을 선택했을 때
                elif (selected_xy_piece.color == whose_turn):
                    if move_from_xy != []:
                        screen_blit_empty_tile(move_from_xy, move_from_win_xy, screen)
                        screen_blit_selected_piece(board.pos(move_from_xy[0], move_from_xy[1]), move_from_win_xy, screen, whose_turn)
                        screen.blit(img_board,(0,0))
                        screen_blit_all_pieces(board, screen, whose_turn, game_state)
                        
                    # 같은 말을 다시 선택했을 때 선택 취소
                    if move_from_xy == selected_xy:
                        move_from_xy, move_from_win_xy = [], []
                    else:
                        # 클릭한 위치의 좌표와 말 저장
                        move_from_xy, move_from_win_xy = selected_xy, selected_win_xy
                        selected_piece = board.board[move_from_xy[1]][move_from_xy[0]] # board.board는 x, y를 뒤집어 인식함
                        
                        # 정보 출력하고, 선택된 기물 그래픽으로 표시
                        screen_blit_selected_piece(selected_piece, move_from_win_xy, screen, whose_turn)
                        show_possible_moves(selected_piece, board, screen)
                
                # 자신의 말이 선택되어 있을 때
                elif move_from_xy != []:
                    # 클릭한 위치의 좌표 저장
                    move_to_xy, move_to_win_xy, move_to_chessboard_xy = selected_xy, selected_win_xy, selected_chessboard_xy
                    
                    # 갈 수 있는 곳을 선택하여 말이 움직인 경우
                    if selected_piece.movable(board, *move_to_xy):
                        is_killable = board.killable(move_from_xy[0], move_from_xy[1], move_to_xy[0], move_to_xy[1])
                        selected_piece.move(board, move_to_xy[0], move_to_xy[1])

                        if not is_muted:
                            # Play capture sound if capturing a piece, otherwise play move sound
                            if is_killable:
                                sounds['capture'].play()
                            else:
                                sounds['place'].play()

                        print_xy(move_to_xy, move_to_chessboard_xy, board)

                        # 폰이 승진할 수 있다면 
                        if (type(selected_piece) == Pawn) and (
                            (((whose_turn == -1) and (move_to_xy[1] == 0)) or (
                            (whose_turn == 1) and (move_to_xy[1] == 7)))):

                            # Clear the board and redraw everything
                            screen.blit(img_board,(0,0))
                            screen_blit_all_pieces(board, screen, whose_turn, game_state)
                            pygame.display.update()

                            # 승진할 기물을 선택하라는 이미지 띄우기
                            screen_blit_about_promotion(move_to_xy, move_to_win_xy, screen, whose_turn)

                            # 폰 승진 단계로 이동
                            promotionable = True

                        # 승진 할 수 없다면 턴 종료
                        else:
                            screen.blit(img_board,(0,0))
                            game_state = gamestate(board, whose_turn)
                            screen_blit_all_pieces(board, screen, whose_turn, game_state)
                            whose_turn *= -1
                            
                            # 체크, 체크메이트, 스테일메이트 여부 확인
                            if game_state == "Check":
                                if not is_muted:
                                    sounds['check'].play()
                                print("Check!")

                            elif game_state == "Checkmate" or game_state == "Stalemate":
                                if not is_muted:
                                    sounds['game_end'].play()
                                    pygame.mixer.music.stop()
                                game_over = True
                                if game_state == "Checkmate":
                                    print("Checkmate!")
                                else:
                                    print("Stalemate!")

                    # 갈 수 없는 곳이나, 선택 했던 곳을 또 선택할 경우 선택 취소
                    else:
                        if selected_piece.movable(board, move_to_xy[0], move_to_xy[1]):
                            if not is_muted:
                                sounds['blocked'].play()
                        screen_blit_empty_tile(move_from_xy, move_from_win_xy, screen)
                        screen_blit_selected_piece(selected_piece, move_from_win_xy, screen, whose_turn)
                        screen.blit(img_board,(0,0))
                        screen_blit_all_pieces(board, screen, whose_turn, game_state)
                    
                    # 턴 전환 혹은 선택 취소
                    if not promotionable:
                        move_from_xy = []
                        move_from_win_xy = []
                        move_to_xy = []
                        move_to_win_xy = []
                        move_to_chessboard_xy = []
                
                # 자신의 말을 선택하지 않았을 때
                else: 
                    move_from_xy = []
                    move_from_win_xy = []

                pygame.display.update()


# 게임 창 띄우기
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption("와! 체스!") # 창 제목

is_muted = False
quit = False

while not quit:
    quit = run_game()

pygame.quit()