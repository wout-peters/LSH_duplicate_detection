from bs4 import BeautifulSoup
import pandas as pd
import requests

def getData(URL):
    brands = pd.DataFrame()

    #Initialize web scraper
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    return None


def main():
    print(getData("https://www.kitele.com/en/tv-brands"))

if __name__ == "__main__":
    main()