import tkinter as tk
from tkinter import ttk
import csv
from admin_panel import *
from pos import *


def read_csv_file():
    with open('users.csv', 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader]


def check_credentials(username, password):
    data = read_csv_file()
    for row in data:
        if row[0] == username and row[1] == password:
            return row[2]
    return False


def login_window():
    login_page = tk.Tk()
    login_page.title("Login Page")
    login_page.geometry("400x300")

    title_label = tk.Label(login_page, text="Login to Your Account",
                           font=("Helvetica", 18, "bold"), pady=20)
    title_label.pack()

    username_frame = tk.Frame(login_page)
    username_frame.pack()

    username_label = tk.Label(username_frame, text="Username:",font=("Helvetica", 15), padx=10, pady=10)
    username_label.pack(side="left")

    username_entry = tk.Entry(username_frame, font=("Helvetica", 15), width=25)
    username_entry.pack(side="right")

    password_frame = tk.Frame(login_page)
    password_frame.pack()

    password_label = tk.Label(password_frame, text="Password:",
                              font=("Helvetica", 15), padx=10, pady=10)
    password_label.pack(side="left")

    password_entry = tk.Entry(password_frame, font=("Helvetica", 15),
                              width=25, show="*")
    password_entry.pack(side="right")

    submit_button = tk.Button(login_page, text="Login",
                              font=("Helvetica", 14, "bold"), width=15,
                              bg="#007bff", fg="black",
                              command=lambda: login(
                                  username_entry.get(), password_entry.get()))
    submit_button.pack(pady=20)

    error_label = tk.Label(login_page, font=("Helvetica", 15), fg="red")

    def login(username, password):
        role = check_credentials(username, password)
        if role == "admin":
            login_page.destroy()
            admin_panel_window()
        elif role == "user":
            login_page.destroy()
            create_pos_system_window()
        else:
            error_label.config(text="Incorrect username or password")
            error_label.pack()

    login_page.mainloop()


login_window()
