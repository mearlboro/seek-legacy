#!/usr/bin/env ruby

files = Dir.glob("../raw/herox/*")
queue = Queue.new
files.map { |file| queue << file }
thread_count = 10
threads = thread_count.times.map do
  Thread.new do
    while !queue.empty?
      elem = queue.pop
      system("python extractor.py #{elem} ../txt")
    end
  end
end
threads.each(&:join)
# files.each_slice(100) do |batch|
#     threads = []
#     batch.each do |elem|
#       threads << Thread.new do
#         system("python extractor.py #{elem}")
#       end
#     threads.each(&:join)
#   end
# end

# files.each do |elem|

# end

# Theana for python
# Torch / Lua
