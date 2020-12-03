=begin
a = [1, 3, 5]

File.open('www', 'w'){ |f1|
  f1.puts 'asdasd'
  f1.puts '1111111111'
  f1.puts a
}
=end

require 'watir'

#add1 = 'https://news.einfomax.co.kr/news/articleView.html?idxno=4073883'

browser1 = Watir::Browser.new :firefox

#browser1.goto add1
#next_page1 = browser1.div(id: 'article-view-content-div').text
#browser1.driver.manage.timeouts.implicit_wait = 3
#p next_page1
#exit

#browser1 = Selenium::WebDriver::Firefox::Profile.new
add1 = 'https://news.einfomax.co.kr/news/articleList.html?sc_area=A&view_type=sm&sc_word=%EC%84%9C%ED%99%98-%EB%A7%88%EA%B0%90'

browser1.goto add1
browser1.driver.manage.timeouts.implicit_wait = 10

#d = browser1.div(class: 'clearfix')

#bb = browser1.element(:xpath => "//a[@class='line-height-3-2x']")
#browser1.elements(:xpath => "//p[@class='list-summary']").each{ |x|
 # p x.a.attribute_value('href')
  #p x.div.text
#}

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

#pages = []
#browser1.elements(:xpath => "//a[@class='line-height-3-2x']").each{ |x|
#  pages << x.attribute_value('href')
#}
#p bb.htmls
#p bb[:href]
#p browser1.links.map(&:href)

#pages = '123456789'.split('')
#pages << "10"

#pages.each{ |x|
#  browser1.link(text: x).click
#}
#(2..3).each{ |x|

#browser1.i(class: 'fa fa-angle-right fa-fw').click
#browser1.element(:xpath => "//i[@class='fa fa-angle-right fa-fw']").click

#page1 = next_page1.attribute_value('href')
#p page1
#browser1.goto page1

t_adds, t_dates = [], []
(1..68).each{ |x|
  if x % 10 != 1                 
    browser1.link(text: x.to_s).click
    browser1.driver.manage.timeouts.implicit_wait = 10
  end
  p x
  adds, dates = get_pages(browser1)
  t_adds += adds
  t_dates += dates

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

#element(:xpath => "//a[@href='line-height-3-2x']").each{ |x|
#  pages << x.attribute_value('href')
#}

#browser1.link(text: '2').click
#p d.text

#p d

#p d.text_field

#p d.html
#p d.htmls
#p '-----------'
#p d.link
#p d.links
#p d.innertext

#Watir::Browser.new :phantomjs

#class 'list-titles'


# division class 'clearfix'
# page
# li


