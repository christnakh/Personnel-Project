import tkinter as tk
import csv
from tkinter import messagebox


def read_csv_file():
    with open('users.csv', 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader]


def write_csv_file(data):
    with open('users.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def add_user_window():
    add_user_page = tk.Toplevel()
    add_user_page.title("Add User")
    add_user_page.geometry("400x250")

    input_frame = tk.Frame(add_user_page)
    input_frame.pack(pady=20)

    username_label = tk.Label(input_frame, text="Username:", font=("Arial", 12))
    username_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    username_entry = tk.Entry(input_frame, font=("Arial", 12))
    username_entry.grid(row=0, column=1, padx=10, pady=5, ipadx=50, ipady=5)

    password_label = tk.Label(input_frame, text="Password:", font=("Arial", 12))
    password_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    password_entry = tk.Entry(input_frame, show="*", font=("Arial", 12))
    password_entry.grid(row=1, column=1, padx=10, pady=5, ipadx=50, ipady=5)

    role_label = tk.Label(input_frame, text="Role:", font=("Arial", 12))
    role_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
    role_entry = tk.Entry(input_frame, font=("Arial", 12))
    role_entry.grid(row=2, column=1, padx=10, pady=5, ipadx=50, ipady=5)

    add_user_button = tk.Button(add_user_page, text="Add User", font=("Arial", 12), bg="#4CAF50", fg="black",
                                command=lambda: add_user(username_entry, password_entry, role_entry))
    add_user_button.pack(pady=20)

    add_user_page.mainloop()


def add_user(username_entry, password_entry, role_entry):
    username = username_entry.get()
    password = password_entry.get()
    role = role_entry.get()
    if not username or not password or not role:
        tk.messagebox.showwarning("Warning", "Please fill in all fields.")
        return
    data = read_csv_file()
    user_exists = False
    for row in data:
        if row[0] == username:
            user_exists = True
            break
    if user_exists:
        tk.messagebox.showwarning("Warning", "User already exists!")
    else:
        data.append([username, password, role])
        write_csv_file(data)
        tk.messagebox.showinfo("Success", "User added successfully!")
        username_entry.delete(0, tk.END)
        password_entry.delete(0, tk.END)
        role_entry.delete(0, tk.END)



def remove_user_window():
    remove_user_page = tk.Toplevel()
    remove_user_page.title("Remove User")
    remove_user_page.geometry("400x200")

    input_frame = tk.Frame(remove_user_page)
    input_frame.pack(pady=20)

    username_label = tk.Label(input_frame, text="Username:", font=("Arial", 12))
    username_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    username_entry = tk.Entry(input_frame, font=("Arial", 12))
    username_entry.grid(row=0, column=1, padx=10, pady=5, ipadx=50, ipady=5)

    remove_user_button = tk.Button(remove_user_page, text="Remove User", font=("Arial", 12), bg="#f44336", fg="black", command=lambda: remove_user(username_entry, username_entry.get()))
    remove_user_button.pack(pady=20)

    remove_user_page.mainloop()

def remove_user(username_entry, username):
    if not username:
        tk.messagebox.showerror("Error", "Please enter a username.")
        return
    
    data = read_csv_file()
    if any(row[0] == username for row in data):
        data = [row for row in data if row[0] != username]
        write_csv_file(data)
        tk.messagebox.showinfo("Success", "User removed successfully!")
        username_entry.delete(0, tk.END)  
    else:
        tk.messagebox.showerror("Error", "User not found.")

def edit_user_window():
    edit_user_page = tk.Toplevel()
    edit_user_page.title("Edit User")
    edit_user_page.geometry("400x300")

    input_frame = tk.Frame(edit_user_page)
    input_frame.pack(pady=20)

    old_username_label = tk.Label(input_frame, text="Old username:", font=("Arial", 12))
    old_username_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    old_username_entry = tk.Entry(input_frame, font=("Arial", 12))
    old_username_entry.grid(row=0, column=1, padx=10, pady=5, ipadx=50, ipady=5)

    username_label = tk.Label(input_frame, text="New username:", font=("Arial", 12))
    username_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    username_entry = tk.Entry(input_frame, font=("Arial", 12))
    username_entry.grid(row=1, column=1, padx=10, pady=5, ipadx=50, ipady=5)

    password_label = tk.Label(input_frame, text="Password:", font=("Arial", 12))
    password_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
    password_entry = tk.Entry(input_frame, font=("Arial", 12))
    password_entry.grid(row=2, column=1, padx=10, pady=5, ipadx=50, ipady=5)

    role_label = tk.Label(input_frame, text="Role:", font=("Arial", 12))
    role_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
    role_entry = tk.Entry(input_frame, font=("Arial", 12))
    role_entry.grid(row=3, column=1, padx=10, pady=5, ipadx=50, ipady=5)

    def save_changes():
        old_username = old_username_entry.get()
        username = username_entry.get()
        password = password_entry.get()
        role = role_entry.get()
        if not old_username or not username or not password or not role:
            tk.messagebox.showwarning("Warning", "Please fill in all fields!")
        else:
            edit_user(old_username, username, password, role)
            old_username_entry.delete(0, 'end')
            username_entry.delete(0, 'end')
            password_entry.delete(0, 'end')
            role_entry.delete(0, 'end')


    edit_user_button = tk.Button(edit_user_page, text="Save", font=("Arial", 12), bg="#4caf50", fg="black", command=save_changes)
    edit_user_button.pack(pady=20)

    edit_user_page.mainloop()

def edit_user(old_username_entry, username_entry, password_entry, role_entry):
    data = read_csv_file()
    old_username_exists = False
    for i, row in enumerate(data):
        if row[0] == old_username_entry:
            data[i] = [username_entry, password_entry, role_entry]
            old_username_exists = True
            break
    if not old_username_exists:
        tk.messagebox.showerror("Error", "User not found!")
    else:
        write_csv_file(data)
        tk.messagebox.showinfo("Success", "User edited successfully!")



class ViewUserWindow(tk.Toplevel):
    def __init__(self, master):
        self.master = master
        self.master.title("View Users")
        self.master.geometry("800x600")

        self.search_frame = tk.Frame(self.master)
        self.search_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.search_label = tk.Label(
            self.search_frame, text="Enter Product Name:")
        self.search_entry = tk.Entry(self.search_frame)
        self.search_button = tk.Button(
            self.search_frame, text="Search/Refresh", command=self.search_users)

        self.search_label.pack(side=tk.LEFT)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_button.pack(side=tk.LEFT, padx=5)

        self.table_frame = tk.Frame(self.master)
        self.table_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.users_table = []
        with open('users.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.users_table.append(row)

        self.display_products()

    def display_products(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        table_header = ["User Name", "Passowrd",
                        "Role"]
        for i, header in enumerate(table_header):
            header_label = tk.Label(self.table_frame, text=header, font=(
                "Arial", 14, "bold"), bg="#f2f2f2", fg="black", padx=10, pady=5, borderwidth=1, relief=tk.SOLID)
            header_label.grid(row=0, column=i, sticky="nsew")

        for i, users in enumerate(self.users_table):
            for j, item in enumerate(users):
                user_label = tk.Label(self.table_frame, text=item, font=(
                    "Arial", 15), padx=15, pady=10, borderwidth=1, relief=tk.SOLID)
                user_label.grid(row=i+1, column=j, sticky="nsew")

    def search_users(self):
        search_term = self.search_entry.get()

        if search_term == "":
            self.users_table = []
            with open('users.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    self.users_table.append(row)
        else:
            matched_users = []
            for users in self.users_table:
                if search_term.lower() in users[0].lower():
                    matched_users.append(users)

            self.users_table = matched_users

        self.display_products()


def view_users_window():
    root = tk.Tk()
    app = ViewUserWindow(root)
    root.mainloop()
