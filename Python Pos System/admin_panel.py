import tkinter as tk
from user import *
from product import *
from pos import *


class AdminPanelWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Admin Panel")
        self.geometry("900x500")

        button_frame = tk.Frame(self, pady=20)
        button_frame.pack(expand=True)

        pos_system_button = tk.Button(
            button_frame, text="Pos System", command=create_pos_system_window, width=20, height=2)
        pos_system_button.grid(row=0, column=0, padx=20, pady=10)

        add_user_button = tk.Button(
            button_frame, text="Add User", command=add_user_window, width=20, height=2)
        add_user_button.grid(row=0, column=1, padx=20, pady=10)

        remove_user_button = tk.Button(
            button_frame, text="Remove User", command=remove_user_window, width=20, height=2)
        remove_user_button.grid(row=0, column=2, padx=20, pady=10)

        edit_user_button = tk.Button(
            button_frame, text="Edit User", command=edit_user_window, width=20, height=2)
        edit_user_button.grid(row=1, column=0, padx=20, pady=10)

        add_product_button = tk.Button(
            button_frame, text="Add Product", command=add_product_window, width=20, height=2)
        add_product_button.grid(row=1, column=1, padx=20, pady=10)

        remove_product_button = tk.Button(
            button_frame, text="Remove Product", command=remove_product_window, width=20, height=2)
        remove_product_button.grid(row=1, column=2, padx=20, pady=10)

        edit_product_button = tk.Button(
            button_frame, text="Edit Product", command=edit_product_window, width=20, height=2)
        edit_product_button.grid(row=2, column=0, padx=20, pady=10)

        view_products_button = tk.Button(
            button_frame, text="View/Search Products", command=view_product_window, width=20, height=2)
        view_products_button.grid(row=2, column=1, padx=20, pady=10)

        search_product_button = tk.Button(
            button_frame, text="View/Search Users", command=view_users_window, width=20, height=2)
        search_product_button.grid(row=2, column=2, padx=20, pady=10)

        sales_of_the_day_button = tk.Button(
            button_frame, text="Sales of the Day", command=edit_user_window, width=20, height=2)
        sales_of_the_day_button.grid(row=3, column=0, padx=20, pady=10)

        sales_of_the_week_button = tk.Button(
            button_frame, text="Sales of the Week", command=edit_user_window, width=20, height=2)
        sales_of_the_week_button.grid(row=3, column=1, padx=20, pady=10)

        sales_of_the_month_button = tk.Button(
            button_frame, text="Sales of the Month", command=edit_user_window, width=20, height=2)
        sales_of_the_month_button.grid(row=3, column=2, padx=20, pady=10)

        sales_of_the_year_button = tk.Button(
            button_frame, text="Sales of the Year", command=edit_user_window, width=20, height=2)
        sales_of_the_year_button.grid(row=4, column=0,    padx=20, pady=10)


        sales_of_the_year_button = tk.Button(
            button_frame, text="Stastic of the Sales", command=statitic_of_the_sales_window, width=20, height=2)
        sales_of_the_year_button.grid(row=4, column=1,    padx=20, pady=10)

        quit_button = tk.Button(
            self, text="Quit", command=self.quit, width=10, height=1)
        quit_button.pack(side=tk.BOTTOM, pady=20)

        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        size = tuple(int(_) for _ in self.geometry().split('+')[0].split('x'))
        x = screen_width/2 - size[0]/2
        y = screen_height/2 - size[1]/2
        self.geometry("+%d+%d" % (x, y))


def admin_panel_window():
    admin_panel = AdminPanelWindow()
    admin_panel.mainloop()
