import scrapy
from pathlib import Path
import json


class WikipediaSpider(scrapy.Spider):
    name = "wikipedia_philosophy"
    allowed_domains = ["wikipedia.org"]
    
    start_urls = [
        "https://en.wikipedia.org/wiki/Category:Philosophy"
    ]

    MAX_ARTICLES = 10000
    article_count = 0
    visited_categories = set()

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.5,
        "AUTOTHROTTLE_ENABLED": True
    }

    def parse(self, response):
        yield from self.parse_category(response)

    def parse_category(self, response):
        # Avoid infinite loops
        if response.url in self.visited_categories:
            return
        self.visited_categories.add(response.url)

        # 1️⃣ Crawl subcategories (if any)
        subcategories = response.css(
            "#mw-subcategories a::attr(href)"
        ).getall()

        for subcat in subcategories:
            if self.article_count >= self.MAX_ARTICLES:
                return
            yield response.follow(subcat, callback=self.parse_category)

        article_links = response.css(
            "#mw-pages a::attr(href)"
        ).getall()

        for link in article_links:
            if self.article_count >= self.MAX_ARTICLES:
                return

            if link.startswith("/wiki/") and ":" not in link:
                yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        if self.article_count >= self.MAX_ARTICLES:
            return
        title = response.url.split("/")[-1]
        # paragraphs = response.css(
        #     "div#mw-content-text p::text"
        # ).getall()
        
        paragraphs = response.xpath(
               "//div[@id='mw-content-text']//p//text()"
           ).getall()
        file_path = Path("wikipedia_philosophy.json")

        
        # p.strip() for p in paragraphs if p.strip()
        content = " ".join(p.strip() for p in paragraphs if p.strip())

        if len(content) < 500:
            return

        self.article_count += 1

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
