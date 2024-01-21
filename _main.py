import tkinter as tk
from tkinter import scrolledtext, Button
import _test_driver as td

def solve_puzzle():
    # solution = td.solution_path
    # print(solution)
    # solution_text.delete(1.0, tk.END)
    # solution_text.insert(tk.INSERT, ' '.join(solution))
    td.start_moving()
    
def start_puzzle_solver():
    solve_button['state'] = 'normal'
    solution_text.delete(1.0, tk.END)

root = tk.Tk()
root.title("Puzzle Challenge")
root.attributes("-topmost", True)

start_button = tk.Button(root, text="Start Solver", command=start_puzzle_solver)
start_button.pack()

solve_button = Button(root, text="Solve Puzzle", state='disabled', command=solve_puzzle)
solve_button.pack()

solution_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
solution_text.pack()

bottom_label = tk.Label(root, text="Powered by Shehan")
bottom_label.pack(side=tk.BOTTOM)

root.mainloop()

