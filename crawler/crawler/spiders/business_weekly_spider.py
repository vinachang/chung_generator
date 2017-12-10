import scrapy
from w3lib.html import remove_tags
# scrapy crawl quotes -o quotes.jl
class ArticleSpider(scrapy.Spider):
    name = "businessweekly"
    start_urls = [
        'http://www.businessweekly.com.tw/blogarticle.aspx?id=0000000003',
    ]

    def parse(self, response):
        for article_link in response.css('article.channelnew>a::attr("href")').extract():
            # print('{}&p=0'.format(article_link))
            yield response.follow('{}&p=0'.format(article_link), self.parse_article)

            # yield {
            #     'text': quote.css('span.text::text').extract_first(),
            #     'author': quote.xpath('span/small/text()').extract_first(),
            # }

        next_page = response.css('.nextpage a::attr("href")').extract_first()
        if next_page is not None:
            yield response.follow(next_page, self.parse)

# request = scrapy.Request("http://www.example.com/some_page.html",
#                              callback=self.parse_page2)
#     request.meta['item'] = item
#     yield request
#
# def parse_page2(self, response):
#     item = response.meta['item']
#     item['other_url'] = response.url
#     yield item

    def parse_article(self, response):
        article = response.css('article')
        date = article.css('.pageIntro .articleDate::text').extract_first().strip()
        title = article.css('.pageIntro .headline>h1::text').extract_first()
        # content = '\n'.join([p + '\n' for p in article.css('.articlebody >p::text').extract()])
        content = '\n'.join([remove_tags(p.replace('<br/>', '\n')) + '\n' for p in article.css('.articlebody >p').extract()])
        yield {
            'link': response.url,
            'date': date,
            'title': title,
            'content': content,
        }
