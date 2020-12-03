=begin
read contents from address
== output
* file with name of date
* contens with column at the date
=end

require 'watir'

#browser1 = Watir::Browser.new :firefox

f1_s = 'address.txt'
f2_s = 'dates.txt'

address, dates, address_dates = [], [], []
File.open(f1_s).each_line{ |x| address << x }
File.open(f2_s).each_line{ |x| dates << x }
address_dates = address.zip dates

address_dates.each{ |add1, date1|
  date1 = date1.strip
  if add1.strip == ''
    next
  end
  browser1.goto add1
  text1 = browser1.div(id: 'article-view-content-div').text
  browser1.driver.manage.timeouts.implicit_wait = 10
  File.open("info_#{date1}", 'w'){ |f1|
    f1.write text1
  }
}

browser1.close
