import chess
from utils.analyze import get_legal_moves_with_action_ids
from IPython.display import SVG, HTML, display
import chess.svg


def draw_numbers_on_board(
    square_to_numbers_map: dict[chess.Square, list[str]],
    board: chess.Board | None = None,
    text_color: str = "red",
    font_size: int = None,
    line_height: str = "1.2em", # <-- Vertical distance between lines (adjust as needed)
    board_size: int = 400,
    **kwargs
) -> str:
    """
    Generates an SVG representation of a chess board with numbers/text drawn on
    specified squares. Displays multiple numbers on separate lines using <tspan>.

    Args:
        square_to_numbers_map: A dictionary where keys are chess.Square objects and
                               values are LISTS of strings (numbers) to draw.
        board: An optional chess.Board object. Defaults to the standard starting position.
        text_color: The color of the text to be drawn.
        font_size: The font size of the text.
        line_height: The vertical spacing between lines ('dy' attribute for <tspan>).
        board_size: The size of the SVG board in pixels.
        **kwargs: Additional keyword arguments to pass to chess.svg.board
                  (e.g., squares, arrows, orientation).

    Returns:
        An SVG string representing the board with the numbers drawn on it.
    """
    if board is None:
        board = chess.Board()

    # Generate the base SVG
    svg_string = chess.svg.board(board=board, size=board_size, **kwargs)
    square_size = chess.svg.SQUARE_SIZE

    all_text_elements = ""

    for square, numbers_to_draw_list in square_to_numbers_map.items():
        if not numbers_to_draw_list:
            continue

        # Calculate SVG coordinates for the *start* of the text block
        file_index = chess.square_file(square)
        rank_index = 7 - chess.square_rank(square)
        # Center horizontally, start text slightly below the vertical center to accommodate multiple lines
        x_coord = file_index * square_size + square_size / 2 - 7 + square_size * 0.5
        # Adjust initial Y based on expected number of lines & font size (heuristic)
        num_lines = len(numbers_to_draw_list)
        font_size = font_size or 20 if num_lines <= 2 else 13 if num_lines <= 3 else 10
        initial_y_offset = (font_size * (num_lines -1) * 0.6) # Approximate adjustment based on line height factor
        y_coord = rank_index * square_size + square_size / 2 - 8 + square_size * 0.5 - initial_y_offset # Start near top

        tspan_elements = ""
        for i, number_str in enumerate(numbers_to_draw_list):
            # Set 'dy' for line spacing, but only for lines after the first
            dy_attr = f'dy="{line_height}"' if i > 0 else ""
            # All tspans need x attribute for horizontal centering
            tspan_elements += f'<tspan x="{x_coord}" {dy_attr}>{number_str}</tspan>'

        # Create the main <text> element wrapping the <tspan>s
        text_element = (
            f'<text x="{x_coord}" y="{y_coord}" font-size="{font_size}" '
            f'fill="{text_color}" text-anchor="middle" dominant-baseline="central">'
            f'{tspan_elements}</text>' # dominant-baseline="central" might work better here
        )
        all_text_elements += text_element

    # Inject all text elements into the SVG
    closing_tag_index = svg_string.rfind('</svg>')
    if closing_tag_index != -1:
        modified_svg = svg_string[:closing_tag_index] + all_text_elements + svg_string[closing_tag_index:]
    else:
        modified_svg = svg_string + all_text_elements

    return SVG(modified_svg)


def display_svgs_horizontally(svg_list: list[str]):
    """
    Displays a list of SVG strings horizontally in a Jupyter cell using CSS Flexbox.

    Args:
        svg_list: A list where each element is an SVG string.
    """
    if not svg_list:
        print("No SVGs provided to display.")
        return

    # Ensure items in the list are strings
    valid_svg_strings = []
    for i, item in enumerate(svg_list):
        if isinstance(item, str):
            valid_svg_strings.append(item)
        else:
            # Attempt to get the SVG string if it's an SVG object
            try:
                # IPython.display.SVG stores data in _repr_svg_() or .data
                if hasattr(item, '_repr_svg_'):
                    svg_data = item._repr_svg_()
                    if isinstance(svg_data, str):
                         valid_svg_strings.append(svg_data)
                         continue # Skip to next item
                elif hasattr(item, 'data') and isinstance(item.data, str):
                     valid_svg_strings.append(item.data)
                     continue # Skip to next item

            except Exception as e:
                 print(f"Warning: Item {i} is not a valid SVG string or SVG object. Skipping. Error: {e}")

            print(f"Warning: Item {i} is not an SVG string ({type(item)}). Skipping.")


    if not valid_svg_strings:
        print("No valid SVG strings found in the list.")
        return

    # Create div items for each SVG string
    svg_items = "".join([f'<div style="display: inline-block; margin-right: 10px;">{svg_str}</div>' for svg_str in valid_svg_strings])

    # Construct the HTML using a flex container
    html_content = f'<div style="display: flex; flex-wrap: wrap;">{svg_items}</div>'

    display(HTML(html_content))


def draw_possible_action_ids_on_board(board: chess.Board) -> str:
    """
    Draws action IDs on a chess board.

    Args:
        board: The current chess.Board object.

    Returns:
        A string representation of the board with action IDs.
    """
    # Get the legal moves with action IDs
    dest_square_to_action_ids = get_legal_moves_with_action_ids(board)

    # Create a new board with the same FEN
    new_board = chess.Board(board.fen())
    
    # Draw the action IDs on the board
    svg = draw_numbers_on_board(dest_square_to_action_ids, new_board)

    return svg


def board_to_svg(board: chess.Board, size: int = 390) -> str:
    return chess.svg.board(
        board=board,
        size=size,
        lastmove=board.peek() if board.move_stack else None,
        check=board.king(board.turn) if board.is_check() else None)
