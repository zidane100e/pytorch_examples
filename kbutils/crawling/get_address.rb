=begin
This file is to obtain address of an specific column of yeonhap news

== result 
address.txt
date.text

=end

require 'watir'

def get_pages(browser1)
  adds, dates = [], []
  browser1.elements(:xpath => "//p[@class='list-summary']").each{ |x|
    adds << x.a.attribute_value('href')
  }
  browser1.elements(:xpath => "//div[@class='list-dated']").each{ |x|
  dates << x.text[-16..-7]
  }
  return adds, dates
end

add1 = 'https://news.einfomax.co.kr/news/articleList.html?sc_area=A&view_type=sm&sc_word=%EC%84%9C%ED%99%98-%EB%A7%88%EA%B0%90'

browser1 = Watir::Browser.new :firefox
browser1.goto add1
browser1.driver.manage.timeouts.implicit_wait = 10

# it has 68 pages
(1..68).each{ |x|
  if x % 10 != 1                 
    browser1.link(text: x.to_s).click
    browser1.driver.manage.timeouts.implicit_wait = 10
  end
  p x
  adds, dates = get_pages(browser1)

  File.open('address.txt', 'a'){ |f1|
    f1.puts adds
  }
  File.open('dates.txt', 'a'){ |f1|
    f1.puts dates
  }

  if x % 10 == 0
    next_page1 = browser1.element(:xpath => "//li[@class='pagination-next']").a
    add3 = next_page1.attribute_value('href')
    browser1.goto add3
    browser1.driver.manage.timeouts.implicit_wait = 10
    p [' next', x]
  end
}

browser1.close

