import tkinter as tk
import csv
from datetime import datetime
from tkinter import messagebox
import os


def read_products_from_file():
    with open("products.csv", "r") as file:
        reader = csv.reader(file)
        products = list(reader)
    product_dict = {}
    for product in products:
        product_dict[product[0]] = float(product[1]), int(
            product[2]), int(product[3]), int(product[4])
    return product_dict


def pos_system_window(pos_system_page, product_dict):
    bought_items = []

    def add_to_cart(product):
        if product_dict[product][1] > 0:
            bought_items.append(product)
            product_dict[product] = (product_dict[product][0], product_dict[product]
                                     [1] - 1, product_dict[product][2], product_dict[product][3] + 1)
            update_receipt()
            with open("products.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for product, (price, quantity, barcode, counter) in product_dict.items():
                    writer.writerow(
                        [product, price, quantity, barcode, counter])
        else:
            messagebox.showwarning(
                "Out of Stock", f"{product} is currently out of stock.")

    def remove_from_cart(product):
        if product in bought_items:
            bought_items.remove(product)
            product_dict[product] = (product_dict[product][0], product_dict[product]
                                     [1] + 1, product_dict[product][2], product_dict[product][3] - 1)
            update_receipt()
            with open("products.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for product, (price, quantity, barcode, counter) in product_dict.items():
                    writer.writerow(
                        [product, price, quantity, barcode, counter])
        else:
            messagebox.showwarning(
                "Not in Cart", f"{product} is not in the cart.")

    def update_receipt():
        receipt_text.delete("1.0", tk.END)
        total_price = 0
        for item in bought_items:
            price = product_dict[item][0]
            total_price += price
            receipt_text.insert(tk.END, f"{item}: ${price:.2f}\n")
        receipt_text.insert(tk.END, f"\nTotal: ${total_price:.2f}")

    def finalize_receipt():
        nonlocal bought_items
        current_time = datetime.now().strftime("%Y-%m-%d")
        file_path = f"receipts/{current_time}.csv"
        mode = "a" if os.path.exists(file_path) else "w"
        with open(file_path, mode) as file:
            writer = csv.writer(file)
            if mode == "w":
                writer.writerow(["Item", "Price"])
            for item in bought_items:
                writer.writerow([item, product_dict[item][0]])
        with open("sales.csv", "a") as sales_file:
            sales_writer = csv.writer(sales_file)
            for item in bought_items:
                sales_writer.writerow(
                    [current_time, item, product_dict[item][0]])
        messagebox.showinfo(
            "Receipt Saved", f"Receipt saved as {current_time}.csv")
        bought_items = []
        update_receipt()

    products_label = tk.Label(
        pos_system_page, text="Available Products:", font=("Helvetica", 16))
    products_label.pack()

    products_frame = tk.Frame(pos_system_page)
    products_frame.pack()

    for product in product_dict.keys():
        product_button = tk.Button(
            products_frame, text=product, command=lambda product=product: add_to_cart(product))
        product_button.pack(side="left")

    receipt_label = tk.Label(
        pos_system_page, text="Receipt:", font=("Helvetica", 16))
    receipt_label.pack()

    receipt_text = tk.Text(pos_system_page, height=10, width=30)
    receipt_text.pack()

    finalize_button = tk.Button(
        pos_system_page, text='finalize receipt', command=finalize_receipt)
    finalize_button.pack()

    remove_button = tk.Button(pos_system_page, text='remove from cart',
                              command=lambda: remove_from_cart(bought_items[-1] if bought_items else ''))
    remove_button.pack()


def create_pos_system_window():
    product_dict = read_products_from_file()
    root = tk.Tk()
    pos_system_window(root, product_dict)
    root.mainloop()
