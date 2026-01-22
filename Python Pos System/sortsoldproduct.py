import tkinter as tk
from tkinter import ttk
import csv
from collections import defaultdict

def most_sold_products(csv_file):
    products_sold = defaultdict(int)
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            product_name = row[0]
            quantity = int(row[2])
            price = float(row[1])
            products_sold[product_name] += quantity * price
    sorted_products = sorted(products_sold.items(), key=lambda x: x[1], reverse=True)
    return sorted_products

def search_products():
    search_text = search_entry.get().lower()
    if not search_text:
        display_products(most_sold_products(sales_file))
        return
    search_results = [(product, total_sold) for product, total_sold in most_sold if search_text in product.lower()]
    display_products(search_results)

def display_products(products):
    results_treeview.delete(*results_treeview.get_children())
    for i, (product, total_sold) in enumerate(products, start=1):
        results_treeview.insert("", "end", values=(i, product, f"${total_sold:.2f}"))

# GUI setup
root = tk.Tk()
root.title("Most Sold Products")

# Style setup
style = ttk.Style(root)
style.theme_use("clam")
style.configure("Treeview", rowheight=30, background="#333", foreground="white", fieldbackground="#333", font=("Arial", 12))
style.map("Treeview", background=[("selected", "#555")], foreground=[("selected", "white")], highlightcolor=[("focus", "white"), ("!focus", "#333")])
style.configure("TLabel", background="#333", foreground="white", font=("Arial", 14))
style.configure("TEntry", background="white", foreground="black", font=("Arial", 14))
style.configure("TButton", background="#008CBA", foreground="white", font=("Arial", 14))
style.map("TButton", background=[("active", "#005F6B")])

# Search box and button
search_frame = ttk.Frame(root, padding=20)
search_frame.pack(fill="x")
search_entry = ttk.Entry(search_frame, font=("Arial", 16))
search_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)
search_button = ttk.Button(search_frame, text="Search", command=search_products)
search_button.pack(side="left", padx=10)

# Results display
results_frame = ttk.Frame(root, padding=20)
results_frame.pack(fill="both", expand=True)
results_treeview = ttk.Treeview(results_frame, columns=("Rank", "Product", "Total Sales"), show="headings")
results_treeview.column("Rank", width=60, anchor="center")
results_treeview.column("Product", width=300)
results_treeview.column("Total Sales", width=200, anchor="center")
results_treeview.heading("Rank", text="Rank")
results_treeview.heading("Product", text="Product")
results_treeview.heading("Total Sales", text="Total Sales")
results_treeview.pack(fill="both", expand=True)

# Load data
sales_file = "products.csv"
most_sold = most_sold_products(sales_file)
display_products(most_sold)

root.mainloop()


