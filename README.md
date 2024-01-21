# puzzle_solvera# Sliding Puzzle Solver Project

## Project Overview
This project involves creating a Python script to automatically solve a sliding puzzle using Selenium for web automation. The script captures the initial state of the puzzle, calculates the solution, and then automates the moves in the web browser.

## Key Components
- **Selenium WebDriver**: Automates web browser interaction.
- **Puzzle Solving Logic**: Algorithm to find the solution to the sliding puzzle.
- **Dynamic XPath Handling**: Captures and uses XPaths of puzzle pieces to interact with the web page.

## Implementation Steps
1. **Setup Selenium**: Install Selenium and set up WebDriver for your browser.
2. **Capture Initial State**: Use Selenium to capture the initial state of the puzzle, including the XPaths of the pieces.
3. **Solve the Puzzle**: Implement an algorithm to solve the puzzle based on its initial state.
4. **Automate Puzzle Solving**: Use the solution to automate the puzzle-solving process in the browser.

## Code Structure
- `_main.py`: Contains the main logic to initialize the WebDriver, capture the initial state, and start the puzzle-solving process.
- `_test_driver.py`: Includes the implementation of the puzzle-solving algorithm and functions to interact with the web page.

## Running the Project
To run the project, execute the `_main.py` script. Ensure that all dependencies, including Selenium and the appropriate WebDriver, are installed and configured correctly.

## Future Enhancements
- **Improved Algorithm**: Enhance the puzzle-solving algorithm for efficiency or to handle more complex puzzles.
- **UI Enhancements**: Add a graphical user interface for easier interaction and visualization.
- **Error Handling**: Implement robust error handling for web automation steps.

---

*This project demonstrates skills in web automation, algorithm design, and problem-solving in Python.*
