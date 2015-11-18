#!/usr/bin/env ruby

# Parse arguments at input level, make difference between file and directory
files = ARGV[0]
dst   = ARGV[1]

if File.file?(files)
    system("python extractor.py #{files} #{dst}")
else
    queue = Queue.new
    allFiles = Dir["#{files}/**/*"]
    allFiles = allFiles.select { |file| File.file?(file) }
    allFiles.map { |file| queue << file }
    thread_count = 5
    threads = thread_count.times.map do
    Thread.new do
        while !queue.empty?
            elem = queue.pop
            system("python extractor.py #{elem} #{dst}")
        end
    end
    end
threads.each(&:join)
end
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
