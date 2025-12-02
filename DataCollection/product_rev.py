import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://books.toscrape.com/catalogue/page-1.html"
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

titles = []
ratings = []

for book in soup.find_all("article", class_="product_pod"):
    title = book.h3.a["title"]
    rating = book.p["class"][1]  # e.g. 'Three', 'Five'
    titles.append(title)
    ratings.append(rating)

df = pd.DataFrame({
    "title": titles,
    "rating": ratings
})

df.to_csv("book_reviews.csv", index=False)

print("Data collected successfully!")
print(df.head())
