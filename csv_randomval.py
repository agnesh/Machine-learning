import csv
import random

rows = 10
columns = ["Col1", "Col2", "Col3", "Col4", "Col5"]

with open("file.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(columns)

    for _ in range(rows):
        row = [random.randint(1, 100) for _ in range(5)]
        writer.writerow(row)

print("CSV created successfully!")
