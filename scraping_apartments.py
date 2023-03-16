import random
from random import randint
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

from fake_useragent import UserAgent
from bs4 import BeautifulSoup

import argparse

import listing_functions




from random import randint
from time import sleep



def get_info_from_listing(driver, apt_outfile, addr_outfile):
    """ Saves to file the information from a single listing's page.

    Saves per-apartment information (such as square footage, number of beds, etc.) to 'apt_outfile'.
    Saves per-address information (such as address, features of the apartment complex) to 'addr_outfile'.

    Listings are either a single unit (e.g. a house for rent) or multi-units (e.g. an 
    apartment complex with multiple apartments available. The page structure of each type is 
    different, so we scrape the info using a different function for each.
    """

    soup = BeautifulSoup(driver.page_source)
    url = driver.current_url
    listing_functions.save_per_location_info(soup, url, addr_outfile)
    if listing_functions.is_single_unit_listing(soup):
        try:
            listing_functions.save_info_from_single_unit_listing(soup, apt_outfile)
        except Exception as e:
            print("Could not extract info from single-unit listing. Got Exception: ", e)
    else:
        try:
            listing_functions.save_info_from_multi_unit_listing(soup, apt_outfile)
        except Exception as e:
            print("Could not extract info from multi-unit listing. Got Exception: ", e)

   

def is_page_last(driver):
    page_range = driver.find_element(By.CLASS_NAME, "pageRange")
    page_range_text = page_range.text
    numbers = [int(w) for w in page_range_text.split() if w.isdigit()]
    assert len(numbers) == 2, "We expect the page range to have two numbers in it, but apparently it doesn't: " + page_range_text
    return numbers[0] == numbers[1]

def click_to_next_page(driver):
    next_page_link = driver.find_element(By.CSS_SELECTOR, "a.next")
    sleep(randint(3,5))
    actions = ActionChains(driver)
    actions.move_to_element_with_offset(next_page_link, randint(1,10), randint(1,10))
    actions.pause(randint(1,3))
    actions.click()
    actions.perform()

# We sleep, pause, and move randomly to evade scraper detection.
def click_on_element(driver, element):
    sleep(randint(1,4))
    actions = ActionChains(driver)
    actions.move_to_element(element)
    actions.move_by_offset(randint(2, 10), randint(2,10))
    actions.pause(2)
    actions.click()
    actions.perform()

def save_all_results_from_page(driver, apt_outfile, addr_outfile):
    apt_outfile.write("\n\n")
    sleep(random.randint(5,9))
    count = 0
    while True:
        links = driver.find_elements(By.CSS_SELECTOR, "div.item.active.us")
        if count >= len(links):
            break
        link = links[count]
        click_on_element(driver, link)
        sleep(random.random() + randint(1,2))
        get_info_from_listing(driver, apt_outfile=apt_outfile, addr_outfile=addr_outfile)
        print("Saved result # " + str(count + 1))
        driver.back()
        sleep(random.random() + randint(1,2))
        count += 1

def save_all_results(driver, apt_filename, addr_filename, max_pages = 1000):
    with open(apt_filename, 'w') as apt_outfile, open(addr_filename, 'w') as addr_outfile:
        page = 1
        while True:
            print("About to save results from page ", page)
            save_all_results_from_page(driver, apt_outfile=apt_outfile, addr_outfile=addr_outfile)
            print("Finished saving results from page ", page)
            if page >= max_pages or is_page_last(driver):
                print("It's the last page!")
                break
            print("Clicking to next page...")
            click_to_next_page(driver)
            page += 1


def create_driver_and_open_url(chromedriver_path, url):
    ua = UserAgent()
    user_agent = ua.random
    opts = Options()
    opts.add_argument("user-agent=" + user_agent)
    driver = webdriver.Chrome(chromedriver_path, options=opts)
    driver.get(url)
    driver.maximize_window()
    return driver


def parse_args():
    parser = argparse.ArgumentParser(prog='scraping_apartments',
                                     description='Scrapes apartments.com for apartment prices and information.')
    parser.add_argument('--url',
                        default = 'https://apartments.com/charleston-sc',
                        help='apartments.com URL.')
    parser.add_argument('--apt_csv',
                        help='CSV file to which per-apartment results will be saved.')
    parser.add_argument('--addr_csv',
                        help='CSV file to which per-address results will be saved.')
    parser.add_argument('--chromedriver',
                        default='/home/bryce/Downloads/chromedriver',
                        help='Path to Selenium Chromedriver file.')
    parser.add_argument('--max_pages',
                        type=int,
                        default=1000,
                        help='Maximum number of apartments.com results pages to scrape.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    driver = create_driver_and_open_url(args.chromedriver, args.url)
    apt_csv_filename = args.apt_csv
    addr_csv_filename = args.addr_csv
    max_pages = args.max_pages
    save_all_results(driver, apt_filename=apt_csv_filename, addr_filename=addr_csv_filename, max_pages = max_pages)


if __name__ == '__main__':
    main()