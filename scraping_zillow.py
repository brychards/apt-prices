from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import json
from bs4 import BeautifulSoup
import re
import time

def get_results_json(driver, outfile):
    source = driver.page_source
    json_start = source.find('"queryState"')
    json_end = source.find('-->', json_start)
    print(json_start, json_end, json_end - json_start)
    json_str = source[json_start:json_end]
    json_str = '{' + json_str
    outfile.write(json_str)
    outfile.write('\n')
    data =json.loads(json_str)
    results = data['cat1']['searchResults']['listResults']
    return results

def is_current_page_last(driver):
    soup = BeautifulSoup(driver.page_source)
    page_links = soup.find_all('a', {'title': re.compile(r'^Page [0-9]+')})
    page_numbers = [int(pl.string) for pl in page_links]
    max_page_number = max(page_numbers)

    def is_current_page(page_link):
        return 'current page' in page_link.get('title').lower()
    
    for pl in page_links:
        if not is_current_page(pl):
            continue
        current_page_number = int(pl.string)
        return current_page_number == max_page_number


def click_to_next_page(driver):
    next_page_path = "//a[@title='Next page']"
    link = driver.find_element(By.XPATH, next_page_path)
    try:
        link.click()
        print("Clicked to next page successfully")
        return True
    except Exception as e:
        print("Exception!!! ", e)
        return False


outfile = open('zillow_data.json', 'w')
driver = webdriver.Chrome('/home/bryce/Downloads/chromedriver')
zillow_url = 'https://www.zillow.com/homes/for_rent/?searchQueryState=%7B%22mapBounds%22%3A%7B%22west%22%3A-80.47355129260251%2C%22east%22%3A-79.71961452502438%2C%22south%22%3A32.61165654999642%2C%22north%22%3A33.08065065275883%7D%2C%22isMapVisible%22%3Atrue%2C%22filterState%22%3A%7B%22fore%22%3A%7B%22value%22%3Afalse%7D%2C%22mf%22%3A%7B%22value%22%3Afalse%7D%2C%22ah%22%3A%7B%22value%22%3Atrue%7D%2C%22auc%22%3A%7B%22value%22%3Afalse%7D%2C%22nc%22%3A%7B%22value%22%3Afalse%7D%2C%22fr%22%3A%7B%22value%22%3Atrue%7D%2C%22land%22%3A%7B%22value%22%3Afalse%7D%2C%22manu%22%3A%7B%22value%22%3Afalse%7D%2C%22fsbo%22%3A%7B%22value%22%3Afalse%7D%2C%22cmsn%22%3A%7B%22value%22%3Afalse%7D%2C%22fsba%22%3A%7B%22value%22%3Afalse%7D%7D%2C%22isListVisible%22%3Atrue%2C%22mapZoom%22%3A11%7D'
driver.get(zillow_url)
driver.maximize_window()  # we do this to try to avoid "element not clickable" errors
json_results = {"results": []}
i = 1
while True:
    print("Trying to get page " + str(i) + " results...")
    current_page_results = get_results_json(driver, outfile)
    print("Page " + str(i) + " results:\n", current_page_results)
    i += 1
    json_results["results"].append(current_page_results)
    this_is_last_page = is_current_page_last(driver)
    if this_is_last_page:
        break
    else:
        time.sleep(5)  # we do this to try to avoid "element not clickable" errors
        click_result = click_to_next_page(driver)


outfile.close()

print(json_results)

json_out = json.dumps(json_results, sort_keys=True, indent=2)
print(json_out)
with open("zillow_results.json", "w") as of_2:
    of_2.write(json_out)