# coding: utf-8
=begin
This file is to obtain address of an specific column of yeonhap news

== result 
address.txt
date.text

=end

require 'watir'

def get_pages(browser1)
  adds = []
  titles = []
  temp = browser1.elements(:xpath => "//li/dl/dt/a")
  temp.each{|x|
    tt = x
    #puts(tt.exists?)
    #puts(tt)
    add = x.href
    title = x.text

    puts [title, add]
    adds << add
    titles << title
  }

  return adds, titles
end

# economy/stock
add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2=258&sid1=101&date=20200809'
# economy/global
add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=101&sid2=262'
# economy/general
add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=101&sid2=263'
# test dataset
# stock
add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=101&sid2=258'
# global
#add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=101&sid2=262'
# economy general
#add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=101&sid2=263'
# economy/finance
## test
add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2=259&sid1=101&date=20200809'
add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2=259&sid1=101&date=20200810'
## content
add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2=259&sid1=101'
# economy/industry
## test
#add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2=261&sid1=101&date=20200809'
#add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2=261&sid1=101&date=20200802'
## content
add1 = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2=261&sid1=101'

browser1 = Watir::Browser.new :firefox
browser1.goto add1
browser1.driver.manage.timeouts.implicit_wait = 3


# it has 68 pages
#(1..109).each{ |x|
#(1..500).each{ |x|
(1..150).each{ |x|
#(8..12).each{ |x|
  #browser1.link(text: x.to_s).click
  #/html/body/div[1]/table/tbody/tr/td[2]/div/div[4]/a[3]
  links = browser1.elements(:xpath => "//a[@class='nclicks(fls.date)']")
  #puts links.exist
  
  links[2].click
  browser1.driver.manage.timeouts.implicit_wait = 3

  sleep 2
    
  adds, titles = get_pages(browser1)
  sleep 2

  
  #File.open('stock_address.txt', 'a'){ |f1| f1.puts adds  }
  #File.open('stock_titles.txt', 'a'){ |f1| f1.puts titles  }
  #File.open('global_address.txt', 'a'){ |f1| f1.puts adds  }
  #File.open('global_titles.txt', 'a'){ |f1| f1.puts titles  }
  #File.open('general_titles.txt', 'a'){ |f1| f1.puts titles  }
  #File.open('general_address.txt', 'a'){ |f1| f1.puts adds  }
  #File.open('finance_address.txt', 'a'){ |f1| f1.puts adds  }
  File.open('industry_address.txt', 'a'){ |f1| f1.puts adds  }
  #File.open('testset/stock_titles.txt', 'a'){ |f1| f1.puts titles  }
  #File.open('testset/stock_address.txt', 'a'){ |f1| f1.puts adds  }
  #File.open('testset/global_titles.txt', 'a'){ |f1| f1.puts titles  }
  #File.open('testset/global_address.txt', 'a'){ |f1| f1.puts adds  }
  #File.open('testset/general_titles.txt', 'a'){ |f1| f1.puts titles  }
  #File.open('testset/general_address.txt', 'a'){ |f1| f1.puts adds  }
  
}

browser1.close

