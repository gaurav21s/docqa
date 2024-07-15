# web_crawl.py

import scrapy
import os 
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from utils.logger import logger


class WebCrawler(scrapy.Spider):
    """
    A web crawler for scraping data from the NVIDIA CUDA documentation website.
    
    This crawler is designed to navigate through the NVIDIA CUDA documentation,
    extracting content from both the main pages and their sub-links up to a 
    specified depth.
    
    Attributes:
        name (str): The name of the spider.
        allowed_domains (list): The domains the spider is allowed to crawl.
        start_urls (list): The initial URLs to start crawling from.
        max_depth (int): The maximum depth of sub-links to follow.
    """

    name = 'nvidia_cuda_spider'
    allowed_domains = ['docs.nvidia.com']
    start_urls = ['https://docs.nvidia.com/cuda/']
    max_depth = 5

    def parse(self, response):
        """
        Parse the response and extract data from the current page.
        
        This method extracts data from the current page, follows links to 
        sub-pages if the maximum depth hasn't been reached, and yields the 
        extracted data.
        
        Args:
            response (scrapy.http.Response): The response to parse.
        
        Yields:
            dict: A dictionary containing the extracted data (url, title, content).
        """
        # Extract data from the current page
        yield self.extract_data(response)

        # Check if we haven't reached the maximum depth
        depth = response.meta.get('depth', 1)
        if depth < self.max_depth:
            # Find all links on the page
            soup = BeautifulSoup(response.body, 'html.parser')
            links = soup.find_all('a', href=True)

            for link in links:
                url = link['href']
                # Ensure the URL is absolute
                absolute_url = urljoin(response.url, url)

                # Check if the URL is within the allowed domain
                if self.allowed_domains[0] in absolute_url:
                    yield scrapy.Request(
                        absolute_url,
                        callback=self.parse,
                        meta={'depth': depth + 1}
                    )

    def extract_data(self, response):
        """
        Extract relevant data from the response.
        
        This method extracts the title and main content from the response.
        
        Args:
            response (scrapy.http.Response): The response to extract data from.
        
        Returns:
            dict: A dictionary containing the extracted data (url, title, content).
        """
        soup = BeautifulSoup(response.body, 'html.parser')
        
        title = soup.title.string if soup.title else ''
        main_content = ''
        main_div = soup.find('div', class_='document')
        if main_div:
            main_content = main_div.get_text(strip=True)

        return {
            'url': response.url,
            'title': title,
            'content': main_content,
        }

def run_spider():
    """
    Run the web crawler and save the results to a JSON file.
    """
    logger.info("Spider is running")
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': 'output.json'
    })

    if not os.path.exists('output.json'):
        process.crawl(WebCrawler)
        process.start()

if __name__ == "__main__":
    run_spider()