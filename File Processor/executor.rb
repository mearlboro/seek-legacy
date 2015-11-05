#!/usr/bin/env ruby

files = Dir.glob("../Scrape/PDF/*")

# files.each_slice(100) do |batch|
#     threads = []
#     batch.each do |elem|
#       threads << Thread.new do
#         system("python extractor.py #{elem}")
#       end
#     threads.each(&:join)
#   end
# end

files.each do |elem|
  system("python extractor.py #{elem}")
end

# Theana for python
# Torch / Lua
