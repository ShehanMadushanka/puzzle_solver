from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
import cv2
import time
import pytesseract


# Set the path to the tesseract executable
# If you're on Windows, it will be something like 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# On Linux, it's usually just 'tesseract'
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Specify the path to chromedriver.exe (downloaded above)
chrome_driver_path = 'chromedriver/chromedriver'

# Set up the Chrome WebDriver
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(chrome_driver_path, options=options)

# Open the webpage with the puzzle
driver.get('https://www.helpfulgames.com/subjects/brain-training/sliding-puzzle.html')

# Wait for the cookies dialog to appear and accept cookies
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, 'css-fm0uhk'))).click()

# XPath for the puzzle select button
xpath_for_puzzle_select_button = "//button[@name='level'][@value='0']"

# Wait for the element to be present in the DOM
puzzle_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath_for_puzzle_select_button)))

# Check if the button is visible, if not, execute JavaScript to click
if puzzle_button.is_displayed():
    puzzle_button.click()
else:
    driver.execute_script("arguments[0].click();", puzzle_button)

time.sleep(3)

# At this point, you can add code to take a screenshot or interact with the puzzle
driver.save_screenshot('full_page_screenshot.png')

# Load the screenshot
screenshot_path = 'full_page_screenshot.png'  # Update this to the path of your screenshot
image = cv2.imread(screenshot_path)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Check if the image has been loaded correctly
if image is None:
    print("Error loading image")
else:
    print("Image loaded successfully")
    
"""This is the part where you need to add code to analyze the screenshot and extract the state matrix of the puzzle."""

#  Load the image of the empty space
empty_space_path = 'empty_space_screenshot.png'  # Update this path
empty_space_img = cv2.imread(empty_space_path, cv2.IMREAD_GRAYSCALE)

# Calculate the standard deviation of the grayscale image
std_dev = np.std(empty_space_img)

# Set the threshold slightly above the standard deviation
# You may need to adjust the multiplier based on the actual variation in your empty tiles
threshold = 25 # for example, 10% above the standard deviation

# Print the standard deviation and the choen threshold
print(f"Standard Deviation of Empty Space: {std_dev}")
print(f"Chosen Threshold: {threshold}")

    
# Coordinates and size of the puzzle grid and pieces
puzzle_x, puzzle_y = 550, 390  # Top-left corner of the puzzle grid
piece_width, piece_height = 430, 322  # Width and height of each puzzle piece
puzzle_width, puzzle_height = 1298, 974  # Width and height of the whole puzzle grid
number_width, number_height = 50, 95  # Width and height of the number region

# Initialize the state matrix and piece positions dictionary
state_matrix = [[0 for _ in range(3)] for _ in range(3)]
piece_positions = {}

# Function to check if a piece is likely to be the empty space
# This function should return True if the given piece_img is the empty space

def is_empty_space(piece_img, threshold):
    # Calculate the standard deviation of the grayscale image
    gray_piece_img = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray_piece_img)
    print(f"Tile Standard Deviation: {std_dev}")  # Diagnostic print
    return std_dev < threshold

# Function to display each piece with its standard deviation
def display_piece_with_std_dev(piece_img, std_dev, title):
    cv2.imshow(title, piece_img)
    print(f"{title} - Standard Deviation: {std_dev}")

# # Analyze each piece in the grid
# piece_number = 1
# for row in range(3):
#     for col in range(3):
#         # Calculate the top-left corner of the current piece
#         x = puzzle_x + col * piece_width
#         y = puzzle_y + row * piece_height
#         # Extract the piece image
#         piece_img = image[y:y+piece_height, x:x+piece_width]
        
#         # Check if this piece is the empty space
#         if is_empty_space(piece_img, threshold):
#             state_matrix[row][col] = 0
#         else:
#             state_matrix[row][col] = piece_number
#             piece_number += 1

# Print the state matrix and piece positions
# print("State Matrix:")
# for row in state_matrix:
#     print(' '.join(str(cell) for cell in row))
    
# # Analyze and display each piece in the grid
# for row in range(3):
#     for col in range(3):
#         x = puzzle_x + col * piece_width
#         y = puzzle_y + row * piece_height
#         piece_img = image[y:y+piece_height, x:x+piece_width]
#         std_dev = np.std(cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY))
#         title = f"Piece {row},{col}"
#         display_piece_with_std_dev(piece_img, std_dev, title)

# # Wait for a key press to close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""This is the part where the number detection code goes. You can use the code from the previous section to extract the number regions from the screenshot."""

from google.cloud import vision
import os
import io
import cv2

# Set the path to the service account key
service_account_key_path = 'service_account.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

# Initialize the Google Vision API client
client = vision.ImageAnnotatorClient()

# Function to preprocess the image to highlight white numbers
def preprocess_for_white_text(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Invert colors to highlight white text
    inverted = cv2.bitwise_not(binary)

    return inverted

# Function to crop the top left corner and perform OCR with Google Vision API
def detect_text(client, cropped_image):
    # Convert the cropped image to bytes
    _, buffer = cv2.imencode('.jpg', cropped_image)
    content = io.BytesIO(buffer).read()

    # Construct an image instance
    image = vision.Image(content=content)

    # Perform text detection on the image
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Extract the first annotation (full description)
    if texts:
        text = texts[0].description.strip()
    else:
        text = None

    return text

# Iterate over each puzzle piece position
for row in range(3):
    for col in range(3):
        # Calculate the top-left corner of the current piece
        x = puzzle_x + col * piece_width
        y = puzzle_y + row * piece_height

        # Crop the full puzzle image to get the current piece
        piece_image = image[y:y + piece_height, x:x + piece_width]

        # Crop the top-left corner where the number is located
        number_region = piece_image[0:number_height, 0:number_width]

        # Detect text from the cropped number region
        detected_number = detect_text(client, number_region)

        # Update the state matrix with the detected number
        if detected_number and detected_number.isdigit():
            state_matrix[row][col] = int(detected_number)
        else:
            state_matrix[row][col] = 0
            
# Iterate over each puzzle piece position to display the number region
# for row in range(3):
#     for col in range(3):
#         # Calculate the top-left corner of the current piece
#         x = puzzle_x + col * piece_width
#         y = puzzle_y + row * piece_height

#         # Crop the full puzzle image to get the current piece's number region
#         number_region = image[y:y + number_height, x:x + number_width]

#         # Display the cropped number region
#         cv2.imshow(f'Number Region {row},{col}', number_region)

# # Wait for a key press to close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Print the state matrix
for row in state_matrix:
    print(' '.join(str(cell) for cell in row))
    
    
"""This is the part where you need to add code to solve the puzzle. You can use the code from the previous section to solve the puzzle."""

import heapq

# The current state of the puzzle
current_state = state_matrix

# The goal state of the puzzle
goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

def get_possible_moves(state):
    moves = []
    row, col = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)

    if row > 0: moves.append('up')
    if row < 2: moves.append('down')
    if col > 0: moves.append('left')
    if col < 2: moves.append('right')

    return moves

def apply_move(state, move):
    new_state = [row[:] for row in state]  # Deep copy of the state
    row, col = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)

    if move == 'up':
        new_state[row][col], new_state[row-1][col] = new_state[row-1][col], new_state[row][col]
    elif move == 'down':
        new_state[row][col], new_state[row+1][col] = new_state[row+1][col], new_state[row][col]
    elif move == 'left':
        new_state[row][col], new_state[row][col-1] = new_state[row][col-1], new_state[row][col]
    elif move == 'right':
        new_state[row][col], new_state[row][col+1] = new_state[row][col+1], new_state[row][col]

    return new_state

def manhattan_distance(state, goal):
    distance = 0
    for r in range(3):
        for c in range(3):
            val = state[r][c]
            if val != 0:
                goal_row, goal_col = next((gr, gc) for gr in range(3) for gc in range(3) if goal[gr][gc] == val)
                distance += abs(r - goal_row) + abs(c - goal_col)
    return distance

# continuation of the A* search algorithm implementation

def a_star_search(start, goal):
    # Define a priority queue and add the initial state
    queue = []
    heapq.heappush(queue, (manhattan_distance(start, goal), start, []))  # (heuristic cost, state, path)

    while queue:
        # Pop the state with the lowest heuristic cost
        cost, state, path = heapq.heappop(queue)

        # Check if the goal state is reached
        if state == goal:
            return path  # Return the path taken to reach the goal state

        # Generate possible moves from the current state
        for move in get_possible_moves(state):
            new_state = apply_move(state, move)
            new_path = path + [move]

            # Calculate the new cost (path length + heuristic)
            new_cost = len(new_path) + manhattan_distance(new_state, goal)

            # Add the new state to the priority queue
            heapq.heappush(queue, (new_cost, new_state, new_path))

    return None  # Return None if no solution is found

# Example usage
solution_path = a_star_search(current_state, goal_state)
# if solution_path:
#     print("Solution Path:", solution_path)
# else:
#     print("No solution found.")
    
# def apply_move(state, move):
#     new_state = [row[:] for row in state]  # Deep copy of the state
#     row, col = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)

#     if move == 'up':
#         new_state[row][col], new_state[row-1][col] = new_state[row-1][col], new_state[row][col]
#     elif move == 'down':
#         new_state[row][col], new_state[row+1][col] = new_state[row+1][col], new_state[row][col]
#     elif move == 'left':
#         new_state[row][col], new_state[row][col-1] = new_state[row][col-1], new_state[row][col]
#     elif move == 'right':
#         new_state[row][col], new_state[row][col+1] = new_state[row][col+1], new_state[row][col]

#     return new_state


# # Apply the moves
# for move in solution_path:
#     current_state = apply_move(current_state, move)

# is_solved = current_state == goal_state
# print("Final State:")
# for row in current_state:
#     print(row)
# print("\nIs the puzzle solved?", is_solved)

# Close the browser
# driver.quit()