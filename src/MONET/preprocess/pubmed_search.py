import argparse
import glob
import itertools
import os
import time
from functools import partial
from pathlib import Path

import pandas as pd
import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from tqdm.contrib.concurrent import process_map


class PubMedDownloader:
    def __init__(self, driver, download_path=None, delay=20):
        if driver == "chrome":
            chrome_options = Options()

            if download_path is not None:
                download_path = os.path.abspath(download_path)
                if not os.path.exists(download_path):
                    os.makedirs(download_path)
                prefs = {
                    "download.default_directory": download_path,
                    "profile.default_content_settings.popups": 0,
                }
                chrome_options.add_experimental_option("prefs", prefs)

            # chrome_options.add_argument("--pref", {
            #     "download.default_directory": "/home/robert/Downloads",
            #     "download.prompt_for_download": False,
            # })
            # chrome_options.add_argument("--disable-extensions")
            # chrome_options.add_argument("--disable-gpu")
            # chrome_options.add_argument("--no-sandbox") # linux only
            # chrome_options.add_argument("--headless")
            # chrome_options.headless = True # also works

            # executable_path = executable_path or "chromedriver"
            self.driver = webdriver.Chrome(options=chrome_options)
        else:
            raise NotImplementedError(f"Driver {driver} not implemented")

        self.delay = delay
        self.last_url = None
        self.vars = {}

    def teardown_method(self, method=None):
        self.driver.quit()

    def search(self, query, year):
        if not query.replace(" ", "").isalnum():
            raise ValueError(f"Query {query} contains non-alphanumeric characters")
        # Step # | name | target | value
        # 1 | open | /?query=melanoma&filter=simsearch2.ffrft&filter=years.2012-2013 |
        url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ','+')}&filter=simsearch2.ffrft&filter=years.{year}-{year}"
        self.driver.get(url)
        self.last_url = url
        # time.sleep()
        # 2 | setWindowSize | 1920x1055 |
        # self.driver.set_window_size(1920, 1055)
        # 3 | click | id=save-results-panel-trigger |
        WebDriverWait(self.driver, self.delay).until(
            expected_conditions.presence_of_element_located((By.ID, "save-results-panel-trigger"))
        )

        if "No results were found" in self.driver.page_source:
            return
        if "Found 1 result for" in self.driver.page_source:
            return

        self.driver.find_element(By.ID, "save-results-panel-trigger").click()
        # 4 | mouseOver | id=save-results-panel-trigger |
        element = self.driver.find_element(By.ID, "save-results-panel-trigger")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).perform()
        # 5 | mouseOut | id=save-results-panel-trigger |
        WebDriverWait(self.driver, self.delay).until(
            expected_conditions.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
        element = self.driver.find_element(By.CSS_SELECTOR, "body")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).perform()
        # 6 | click | id=save-action-selection |
        WebDriverWait(self.driver, self.delay).until(
            expected_conditions.presence_of_element_located((By.ID, "save-action-selection"))
        )
        self.driver.find_element(By.ID, "save-action-selection").click()
        # 7 | select | id=save-action-selection | label=All results
        dropdown = self.driver.find_element(By.ID, "save-action-selection")
        WebDriverWait(self.driver, self.delay).until(
            expected_conditions.presence_of_element_located(
                (By.XPATH, "//option[. = 'All results']")
            )
        )
        dropdown.find_element(By.XPATH, "//option[. = 'All results']").click()
        # 8 | click | id=save-action-format |
        WebDriverWait(self.driver, self.delay).until(
            expected_conditions.presence_of_element_located((By.ID, "save-action-format"))
        )
        self.driver.find_element(By.ID, "save-action-format").click()
        # 9 | click | id=save-action-format |
        self.driver.find_element(By.ID, "save-action-format").click()
        # 10 | select | id=save-action-format | label=CSV
        dropdown = self.driver.find_element(By.ID, "save-action-format")
        dropdown.find_element(By.XPATH, "//option[. = 'CSV']").click()
        # 11 | click | css=#save-action-panel-form .action-panel-submit |
        WebDriverWait(self.driver, self.delay).until(
            expected_conditions.presence_of_element_located(
                (By.CSS_SELECTOR, "#save-action-panel-form .action-panel-submit")
            )
        )
        self.driver.find_element(
            By.CSS_SELECTOR, "#save-action-panel-form .action-panel-submit"
        ).click()


def search_pubmed(query_list, year, download_path, driver="chrome"):
    # print(len(query_list), query_list, year, download_path, driver)
    pubmed_downloader = PubMedDownloader(driver, download_path=download_path)
    for query in query_list:
        for _ in range(5):
            try:
                time.sleep(1)
                pubmed_downloader.search(query=query, year=year)
            except BaseException as e:
                print(e)
                print(pubmed_downloader.last_url)
                # pubmed_downloader.teardown_method()
                pubmed_downloader = PubMedDownloader(driver, download_path=download_path)
                continue
            else:
                break
    time.sleep(20)


def remove_file(path_output, query_list):
    for idx, query in enumerate(query_list):
        if os.path.exists(str(path_output / f"csv-{query.replace(' ','')[:10]}-set.csv")):
            os.remove(str(path_output / f"csv-{query.replace(' ','')[:10]}-set.csv"))


def rename_file(path_output, query_list, year):
    for idx, query in enumerate(query_list):
        if os.path.exists(str(path_output / f"csv-{query.replace(' ','')[:10]}-set.csv")):
            os.rename(
                str(path_output / f"csv-{query.replace(' ','')[:10]}-set.csv"),
                str(path_output / f"csv-{query.replace(' ','')[:10]}-set-{year}.csv"),
            )


def check_file(path_output, query_list, year):
    downloaded_queries = []
    missing_queries = []

    for idx, query in enumerate(query_list):
        if os.path.exists(str(path_output / f"csv-{query.replace(' ','')[:10]}-set-{year}.csv")):
            downloaded_queries.append(query)
        else:
            missing_queries.append(query)
    return downloaded_queries, missing_queries


if __name__ == "__main__":

    # fitz = pd.read_csv("data/fitzpatrick17k/fitzpatrick17k.csv")
    # skincon = pd.read_csv("data/fitzpatrick17k/annotations_fitzpatrick17k.csv")
    # pd.Series(
    #     sorted(
    #         list(
    #             set(
    #                 fitz["label"].str.lower().to_list()
    #                 + skincon.columns[2:-1].str.lower().to_list()
    #             )
    #         )
    #     )
    # ).to_csv("data/pubmed/labels.csv", index=False, header=False)
    # dermatology
    # melanoma
    # skin cancer
    # skin disease

    parser = argparse.ArgumentParser(
        prog="pubmed_search.py",
        description="search pubmed data",
        epilog="",
    )

    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    parser.add_argument("-q", "--query-file", type=str, help="search queries", required=True)
    parser.add_argument("-y1", "--start-year", type=int, help="search year from", required=True)
    parser.add_argument("-y2", "--end-year", type=int, help="search year to", required=True)

    parser.add_argument("-t", "--thread", type=int, help="thread", required=False, default=1)

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_output = Path(args.output)
    query_file = Path(args.query_file)
    start_year = args.start_year
    end_year = args.end_year
    num_thread = args.thread

    query_list = query_file.read_text().splitlines()
    print(f"Query list: {len(query_list)}")

    for year in range(end_year, start_year - 1, -1):
        # remove file
        remove_file(path_output, query_list)

        # split queries
        downloaded_queries, missing_queries = check_file(path_output, query_list, year)

        print(f"Downloaded queries: {len(downloaded_queries)}")
        print(f"Missing queries: {len(missing_queries)}")

    if num_thread > 1:
        for year in range(end_year, start_year - 1, -1):
            # remove file
            remove_file(path_output, query_list)

            # split queries
            downloaded_queries, missing_queries = check_file(path_output, query_list, year)

            print(f"Downloaded queries: {len(downloaded_queries)}")
            print(f"Missing queries: {len(missing_queries)}")

            missing_queries_split = [
                missing_queries[i * (16) : min((i + 1) * (16), len(missing_queries))]
                for i in range(len(missing_queries) // 16 + 1)
            ]
            assert set(itertools.chain(*missing_queries_split)) == set(missing_queries)

            # run in parallel
            search_pubmed_year = partial(
                search_pubmed,
                year=year,
                download_path=str(path_output),
            )
            process_map(
                search_pubmed_year,
                missing_queries_split,
                max_workers=num_thread,
            )
            # rename downloaded files
            rename_file(path_output, query_list, year)

    else:
        pubmed_downloader = PubMedDownloader("chrome", download_path=str(path_output))
        for query in tqdm.tqdm(query_list):
            # for year in range(start_year, end_year + 1):
            for year in range(end_year, start_year - 1, -1):
                for i in range(5):
                    try:
                        time.sleep(1)
                        pubmed_downloader.search(query=query, year=year)
                    except BaseException as e:
                        print(e)
                        print(f"Error in {query} {year}")
                        continue
                    else:
                        break

    all_df = pd.concat([pd.read_csv(file) for file in glob.glob(str(path_output / "csv-*.csv"))])

    all_df = all_df.drop_duplicates(subset=["PMID"], keep=False)
    all_df.to_csv(str(path_output / "all.csv"), index=False)
