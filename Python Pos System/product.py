import tkinter as tk
import csv
from tkinter import messagebox
import random


def generate_barcode():
    barcode = ''.join(random.choices('0123456789', k=8))
    return barcode


def add_product_window():
    add_item_page = tk.Tk()
    add_item_page.title("Add Item Page")
    add_item_page.geometry("400x250")

    form_frame = tk.Frame(add_item_page, bd=2,relief=tk.RIDGE, padx=10, pady=10)
    form_frame.pack(fill=tk.BOTH, expand=True)


    name_label = tk.Label(form_frame, text="Product Name",font=("Helvetica", 14))
    name_label.grid(row=0, column=0,pady=5)
    product_name_entry = tk.Entry(form_frame, font=("Helvetica", 14), bd=2, relief=tk.SOLID)
    product_name_entry.grid(row=0, column=1, pady=5, padx=10, ipady=5, ipadx=10)

    price_label = tk.Label(form_frame, text="Price", font=("Helvetica", 14))
    price_label.grid(row=1, column=0,pady=5)
    price_entry = tk.Entry(form_frame, font=("Helvetica", 14), bd=2, relief=tk.SOLID)
    price_entry.grid(row=1, column=1, pady=5, padx=10, ipady=5, ipadx=10)

    quantity_label = tk.Label(form_frame, text="Quantity", font=("Helvetica", 14))
    quantity_label.grid(row=2, column=0,pady=5)
    quantity_entry = tk.Entry(form_frame, font=("Helvetica", 14), bd=2, relief=tk.SOLID)
    quantity_entry.grid(row=2, column=1, pady=5, padx=10, ipady=5, ipadx=10)

    def add_product_callback():
        if not product_name_entry.get() or not price_entry.get() or not quantity_entry.get():
            tk.messagebox.showerror("Error", "Please fill all fields before adding a product.")
            return

        add_product(product_name_entry.get(), price_entry.get(), quantity_entry.get())

        product_name_entry.delete(0, tk.END)
        price_entry.delete(0, tk.END)
        quantity_entry.delete(0, tk.END)


    add_button = tk.Button(form_frame, text="Add", font=("Helvetica", 14), bg="#4CAF50", fg="black", command=add_product_callback)
    add_button.grid(row=3, column=1, pady=10, ipady=5, ipadx=10)

    add_item_page.eval('tk::PlaceWindow %s center' %add_item_page.winfo_toplevel())

    add_item_page.mainloop()


def add_product(product_name, price, quantity):
    barcode = generate_barcode()

    product_exists = False
    with open('products.csv', 'r') as file:
        reader = csv.reader(file)
        products = [row for row in reader]

    for i, row in enumerate(products):
        if row[0] == product_name:
            products[i][1] = str(float(row[1]) + float(price))
            products[i][2] = str(int(row[2]) + int(quantity))
            product_exists = True
            tk.messagebox.showinfo("Success", "Added product quantity")
            break

    if not product_exists:
        counter = 0
        products.append([product_name, price, quantity, barcode, counter])
        tk.messagebox.showinfo("Success", "Product added successfully!")

    with open('products.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in products:
            writer.writerow(row)


def remove_product_window():
    remove_product_page = tk.Tk()
    remove_product_page.title("Remove Product")
    remove_product_page.geometry("400x250")

    fg_color = "white"
    btn_color = "#4caf50"

    label_style = {"font": ("Helvetica", 14), "fg": fg_color,"pady": 10}
    entry_style = {"font": ("Helvetica", 14), "fg": fg_color,"width": 30}
    button_style = {"font": ("Helvetica", 14), "fg": "black", "bg": btn_color, "padx": 10, "pady": 5}


    product_label = tk.Label(remove_product_page, text="Product Name:", **label_style)
    product_label.pack()
    product_entry = tk.Entry(remove_product_page, **entry_style)
    product_entry.pack()

    def remove_product_handler():
        product_name = product_entry.get()
        if product_name.strip() == "":
            tk.messagebox.showinfo("Error", "Please enter a product name!")
            return
        remove_product(product_name)
        product_entry.delete(0, tk.END)

        
    remove_button = tk.Button(remove_product_page, text="Remove", command=remove_product_handler, **button_style)
    remove_button.pack()

    remove_product_page.mainloop()



def remove_product(product_name):
    product_exists = False
    with open('products.csv', 'r') as file:
        reader = csv.reader(file)
        products = [row for row in reader]

    for i, row in enumerate(products):
        if row[0] == product_name:
            products.pop(i)
            product_exists = True
            tk.messagebox.showinfo("Success", "Product removed successfully!")
            break

    if not product_exists:
        tk.messagebox.showinfo("Error", "Product does not exist!")

    with open('products.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in products:
            writer.writerow(row)



def edit_product_window():
    edit_item_page = tk.Tk()
    edit_item_page.title("Edit Item Page")
    edit_item_page.geometry("300x300")

    edit_choice = tk.StringVar(value="Both")

    edit_option_label = tk.Label(
        edit_item_page, text="What would you like to edit?")
    edit_option_label.pack()

    both_option = tk.Radiobutton(
        edit_item_page, text="Both", variable=edit_choice, value="Both")
    both_option.pack()

    quantity_option = tk.Radiobutton(
        edit_item_page, text="Quantity", variable=edit_choice, value="Quantity")
    quantity_option.pack()

    price_option = tk.Radiobutton(
        edit_item_page, text="Price", variable=edit_choice, value="Price")
    price_option.pack()

    confirm_button = tk.Button(edit_item_page, text="Confirm", command=lambda: open_edit_page(
        edit_choice.get(), edit_item_page))
    confirm_button.pack()

    edit_item_page.mainloop()


def open_edit_page(edit_choice, edit_item_page):
    edit_item_page.destroy()

    edit_page = tk.Tk()
    edit_page.title("Edit Item")
    edit_page.geometry("300x300")

    name_label = tk.Label(edit_page, text="Product Name")
    name_label.pack()
    product_name_entry = tk.Entry(edit_page)
    product_name_entry.pack()

    if edit_choice == "Both":
        price_label = tk.Label(edit_page, text="Price")
        price_label.pack()
        price_entry = tk.Entry(edit_page)
        price_entry.pack()

        quantity_label = tk.Label(edit_page, text="Quantity")
        quantity_label.pack()
        quantity_entry = tk.Entry(edit_page)
        quantity_entry.pack()

        edit_button = tk.Button(edit_page, text="Edit", command=lambda: edit_product(
            product_name_entry.get(), price_entry.get(), quantity_entry.get(), edit_page))
        edit_button.pack()

    elif edit_choice == "Quantity":
        quantity_label = tk.Label(edit_page, text="Quantity")
        quantity_label.pack()
        quantity_entry = tk.Entry(edit_page)
        quantity_entry.pack()

        edit_button = tk.Button(edit_page, text="Edit", command=lambda: edit_product(
            product_name_entry.get(), None, quantity_entry.get(), edit_page))
        edit_button.pack()

    elif edit_choice == "Price":
        price_label = tk.Label(edit_page, text="Price")
        price_label.pack()
        price_entry = tk.Entry(edit_page)
        price_entry.pack()

        edit_button = tk.Button(edit_page, text="Edit", command=lambda: edit_product(
            product_name_entry.get(), price_entry.get(), None, edit_page))
        edit_button.pack()

    def validate_entry(entry):
        if not entry.get():
            tk.messagebox.showerror("Error", "Product name cannot be empty!")
            return False
        return True

    def on_edit_button_click():
        if validate_entry(product_name_entry):
            edit_product(product_name_entry.get(), price_entry.get(), quantity_entry.get(), edit_page)
            # clear the entry widgets after edit is performed
            product_name_entry.delete(0, tk.END)
            price_entry.delete(0, tk.END)
            quantity_entry.delete(0, tk.END)

    edit_button.config(command=on_edit_button_click)

    edit_page.mainloop()


def edit_product(product_name, price, quantity, edit_page):
    products = []
    with open('products.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            products.append(row)

    product_exists = False
    for i, product in enumerate(products):
        if product[0] == product_name:
            product_exists = True
            if price is not None:
                products[i][1] = price
            if quantity is not None:
                products[i][2] = quantity

    if product_exists:
        with open('products.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for product in products:
                writer.writerow(product)
            tk.messagebox.showinfo("Success", "Product edited successfully!")
    else:
        tk.messagebox.showerror("Error", "Product does not exist")


class ViewProductsWindow(tk.Toplevel):
    def __init__(self, master):
        self.master = master
        self.master.title("View Products")
        self.master.geometry("800x600")

        self.search_frame = tk.Frame(self.master)
        self.search_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.search_label = tk.Label(
            self.search_frame, text="Enter Product Name:")
        self.search_entry = tk.Entry(self.search_frame)
        self.search_button = tk.Button(
            self.search_frame, text="Search/Refresh", command=self.search_products)

        self.search_label.pack(side=tk.LEFT)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_button.pack(side=tk.LEFT, padx=5)

        self.table_frame = tk.Frame(self.master)
        self.table_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.products_table = []
        with open('products.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.products_table.append(row)

        self.display_products()

    def display_products(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        table_header = ["Product Name", "Price",
                        "Quantity", "Barcode", "Number Sold"]
        for i, header in enumerate(table_header):
            header_label = tk.Label(self.table_frame, text=header, font=(
                "Arial", 14, "bold"), bg="#f2f2f2", fg="black", padx=10, pady=5, borderwidth=1, relief=tk.SOLID)
            header_label.grid(row=0, column=i, sticky="nsew")

        for i, product in enumerate(self.products_table):
            for j, item in enumerate(product):
                product_label = tk.Label(self.table_frame, text=item, font=(
                    "Arial", 15), padx=15, pady=10, borderwidth=1, relief=tk.SOLID)
                product_label.grid(row=i+1, column=j, sticky="nsew")

    def search_products(self):
        search_term = self.search_entry.get()

        if search_term == "":
            self.products_table = []
            with open('products.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    self.products_table.append(row)
        else:
            matched_products = []
            for product in self.products_table:
                if search_term.lower() in product[0].lower():
                    matched_products.append(product)

            self.products_table = matched_products

        self.display_products()


def view_product_window():
    root = tk.Tk()
    app = ViewProductsWindow(root)
    root.mainloop()
