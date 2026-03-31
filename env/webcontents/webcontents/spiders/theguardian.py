import scrapy
from pathlib import Path
import json


class TheguardianSpider(scrapy.Spider):
    name = "theguardian"
    allowed_domains = ["theguardian.com"]
    start_urls = ["https://www.theguardian.com/world/2025/jan/01"]

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 2,
        "CONCURRENT_REQUESTS": 4
    }



    # async def start(self):
    #     base_url_world = "https://www.theguardian.com/world"
    #     europe = "europe-news"
    #     base_url_usa = "https://www.theguardian.com/us-news"
    #     south_americas = "americas"
    #     base_url_australia = "https://www.theguardian.com/australia-news"
    #     base_url_tech = "https://www.theguardian.com/uk/technology"
    #     base_url_tech = "https://www.theguardian.com/uk/business"
    #     urls = [
    #         "https://www.theguardian.com/world/2026/jan/28/odesa-russia-crosshairs-war-pivots-back-to-black-sea",
    #     ]
    #     for url in urls:
    #         yield scrapy.Request(url=url, callback=self.parse)
        
        # print(title,paragraphs)
    def parse(self, response):
        #Extract article links
        # news | group-0 | card-@2 | media-picture
        article_links = response.css(
            "a[data-link-name='article']::attr(href)"
        ).getall()

        for link in article_links:
            yield response.follow(link, callback=self.parse_article)

        #Pagination (next page)
        next_page = response.css("a[rel='next']::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)


    def parse_article(self, response):
        title = response.url.split("/")[-1]
        # title = response.css("h3.dcr-1aqe7zu::text").get()
        paragraphs = response.css("div#maincontent p::text").getall()
        # print(paragraphs)


        file_path = Path("theguardian_2025.json")

        
        # p.strip() for p in paragraphs if p.strip()
        content = " ".join(p for p in paragraphs)

        contents = {
            "title": title,
            "url": response.url,
            "content": content
        }

        if file_path.exists() and file_path.stat().st_size > 0:
           with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
             data = []


    #Append new item
        data.append(contents)

    # Write back as a valid JSON array
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)