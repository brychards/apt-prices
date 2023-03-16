"""
This file contains the functions we use to extract a single listing's info from apartments.com.
A listing is a single result in a search for apartments. It can be either a single apartment
(for instance, a single family house for rent) or multiple apartments that are all in the same building,
such as an apartment complex.

The functions take a BeautifulSoup object, and some take an outfile to save the results to - one apartment's
info is saved to a single line in a CSV.
"""

from bs4 import BeautifulSoup
import re

# For each apartment, we will write its info in this format:
PRINT_STR = "{addr};; ${rent};; {beds} bd;; {baths} ba;; {sqft} sq ft\n"
PRICING_GRID_ITEM = "pricingGridItem"
SCREEN_READER_ONLY = "screenReaderOnly"

def get_address_string(soup):
    address_divs = soup.find_all("div", class_="propertyAddressContainer")
    if (len(address_divs) == 0) :
        print("COULD NOT FIND ADDRESS")
        return "UNKNOWN ADDRESS"
    addr_string = " ".join(address_divs[0].text.split())

    # Some properties have a property name like "The Meadows". In these properties, the addr_string contains the street address.
    # For single houses and such, the property name is the street address, and the addr_string will be missing it.
    property_name_split = soup.find("h1", class_="propertyName").text.split()
    if len(property_name_split) == 0:
        print("WARNING: no property name found.")
        return addr_string
    # If the property name starts with a number, we'll assume it's an address.
    if property_name_split[0].isdigit():
        street_address = " ".join(property_name_split)
        addr_string = street_address + ", " + addr_string

    return addr_string

def is_single_unit_listing(soup):
    # Does this page have a div of class "pricingGridItem"? If so, return false. If not, true
    pricing_grid_items = soup.find_all("div", class_=PRICING_GRID_ITEM)
    if (pricing_grid_items):
        return False
    return True

def get_info_from_single_unit_listing(soup, outfile):
    address = get_address_string(soup)
    rent_details = soup.find_all("p", class_="rentInfoDetail")
    printstr = address + ";; " + ";; ".join(map(lambda rd : rd.string, rent_details)) + "\n"
    outfile.write(printstr)

"""
Some multi-unit listings have an exact price row for every unit that's available.
Others just have a price range for each floor plan, e.g. "$1750 - $1900".
This function handles the latter case. We'll just save the price range as the price.
When doing data analysis, we can decide how to handle these cases."""
def get_less_precise_info_from_multi_unit_listing(soup, address, outfile):
    pricing_grid_items = soup.find_all("div", class_=PRICING_GRID_ITEM)
    for item in pricing_grid_items:
        classes = item.parent["class"]
        if 'active' not in classes:
            continue
        rent_range = item.find("span", class_="rentLabel").text.strip()
        ### print("rent range: ", rent_range)
        # The first detailsTextWrapper has the bed, bath, sq ft info
        other_info = item.find(class_="detailsTextWrapper").text
        other_info = other_info.replace("bed", "bd").replace("bath", "ba")
        formatted_info = ";; ".join([w.strip() for w in other_info.split(",")])
        println = "{address};; {rent_range};; {bd_ba_sqft}\n".format(address=address, rent_range=rent_range, bd_ba_sqft=formatted_info)
        outfile.write(println)


def get_info_from_multi_unit_listing(soup, outfile):
    addr = get_address_string(soup)

    pricing_grid_items = soup.find_all("div", class_=PRICING_GRID_ITEM)
    for item in pricing_grid_items:
        classes = item.parent["class"]

        # This excludes not available listings and other listings that aren't shown to the user.
        if 'active' not in classes:
            continue
 
        data_lis = item.find_all("li", attrs={"data-beds" : re.compile(r'.*') })
        
        # Ah ha! This is one of those pages with just a price range
        if len(data_lis) == 0:
            get_less_precise_info_from_multi_unit_listing(soup, addr, outfile)
            return

        BEDS_ATTR = "data-beds"
        BATHS_ATTR = "data-baths"
        for i in range(len(data_lis)):
            li = data_lis[i]
            # The number of beds and baths are attributes of the <li>
            beds = li[BEDS_ATTR]
            baths = li[BATHS_ATTR]

            #Get price
            price_column = li.find("div", class_="pricingColumn")
            price_spans = price_column.find_all("span")
            assert len(price_spans) == 2, "Expected two spans in price column"
            price = price_spans[1].text.strip()

            
            # Get square footage
            sqft_column = li.find("div", class_="sqftColumn")
            sqft_text = sqft_column.text
            sqft_list = map(lambda s: s.replace(",", ""), sqft_text.split())
            sqft_list = [s for s in sqft_list if s.isdigit()]
            sqft = -1
            assert len(sqft_list) == 1, "UNEXPECTED square footage: " + sqft_text
            sqft = sqft_list[0]
            printstr = PRINT_STR.format(addr=addr, rent=price, beds=beds, baths=baths, sqft=sqft)
            outfile.write(printstr)    
