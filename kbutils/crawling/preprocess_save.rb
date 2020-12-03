# coding: utf-8

#f1_s = 'fx_end_news/info_2011-12-14'
dir1_s = 'fx_end_news'
Dir.chdir(dir1_s)
files = Dir.glob("info*")

def separate_div(text0)
  special_tokens = /[▲◇]/
  text1 = text0.split special_tokens
  text1[1] = text1[1].sub(/^.*일 전망.*=*/, '')
  text1[2] = text1[2].sub(/^.*장중 동향.*=*/, '')
  text2 = {:brief => text1[0], :forecast => text1[1], :move => text1[2]}
end

def remove_non_content_from(f1_s)
  File.open(f1_s){ |f1|
    text0 = f1.read
    text0 = text0.sub(/\(서울=연합인포맥스\).*\W+기자.*=/, '')
    match1 = /\w+@[\w\.]+$/ =~ text0
    text0[0...match1]
  }
 end

def arrange(text1)
  text2 = text1.split
  text2.join(' ')
end
=begin
coll = {}
files.each{ |f1_s|
  text1 = remove_non_content_from f1_s
  texts = separate_div text1
  texts.each{ |key, text2|
    texts[key] = arrange text2
  }
  coll[f1_s.to_sym] = texts
}

File.open('preprocessed.dump', 'w'){ |f1|
  Marshal.dump(coll, f1)
}
=end
coll2 = nil
File.open('preprocessed.dump', 'r'){ |f1|
  coll2 = Marshal.load(f1)
}

puts coll2[:'info_2020-03-03'][:brief]
puts
puts coll2[:'info_2020-03-03'][:forecast]
puts
puts coll2[:'info_2020-03-03'][:move]
puts

=begin
write to a csv file
File.open('preprocessed.csv', 'w'){ |f1|
  coll2.each{ |date1, val_dic1|
    col0 = date1.to_s[5..-1]
    col1 = val_dic1[:brief]
    col2 = val_dic1[:forecast]
    col3 = val_dic1[:move]
    str = [col0, col1, col2, col3].join("\t\t")
    f1.write str + "\n"
  }
}
=end

# get statistics
n_briefs, n_forecasts, n_moves = [], [], []
coll2.each{ |date1, val_dic1|
  n_briefs << val_dic1[:brief].split.size
  n_forecasts << val_dic1[:forecast].split.size
  n_moves << val_dic1[:move].split.size
}

puts n_briefs[0..5]
puts n_briefs[0].class

mean1 = n_briefs.sum.to_f / n_briefs.size
mean2 = n_forecasts.sum.to_f / n_briefs.size
mean3 = n_moves.sum.to_f / n_briefs.size

max1 = n_briefs.max
max2 = n_forecasts.max
max3 = n_moves.max

puts [mean1, mean2, mean3]
puts [max1, max2, max3]
