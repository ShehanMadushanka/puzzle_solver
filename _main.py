import tkinter as tk
from tkinter import scrolledtext, Button
import _test_driver as td

def solve_puzzle():
    solution = td.solution_path  # Call your puzzle-solving function
    print(solution)  # Print the solution to the console
    solution_text.delete(1.0, tk.END)  # Clear previous solution
    solution_text.insert(tk.INSERT, ' '.join(solution))  # Display the solution
    
def start_puzzle_solver():
    # _test_driver()  # This should handle the solving and return the solution
    solve_button['state'] = 'normal'  # Enable the solve button
    solution_text.delete(1.0, tk.END)  # Clear previous solution
    # solution_text.insert(tk.INSERT, ' '.join(solution))  # Display the solution

# Create the main window
root = tk.Tk()
root.title("Puzzle Challenge")
root.attributes("-topmost", True)

# Create a button to start the puzzle solver
start_button = tk.Button(root, text="Start Solver", command=start_puzzle_solver)
start_button.pack()

# Create a button to solve the puzzle (disabled initially)
solve_button = Button(root, text="Solve Puzzle", state='disabled', command=solve_puzzle)
solve_button.pack()

# Create a text area to display the solution
solution_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
solution_text.pack()

# Add a bottom label "Powered by Shehan"
bottom_label = tk.Label(root, text="Powered by Shehan")
bottom_label.pack(side=tk.BOTTOM)

# Start the GUI event loop
root.mainloop()

