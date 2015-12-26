#!/usr/bin/env ruby
# begin
#  gem 'mechanize', ">=2.7"
# rescue Gem::LoadError => e
#  system("sudo gem install mechanize")
#  Gem.clear_paths
# end

require 'open-uri'
require 'mechanize'
require 'nokogiri'
require 'net/http'
require 'io/console'
require 'optparse'
require './spiral'

# Defines the number of threads to run Scrapy on
$thread_count = 5

# Downloads array of files into the target_dir according to the file_names
# Current link is used in the case of relative URIs to create the full path
def download_files_from_URLs(agent, target_dir, links, override, file_in_names, current_link)
  # Open file from web URL, using username and password provided

  uri = URI(current_link)

  queue = Queue.new
  name_queue = Queue.new

  links.map { |url| queue << url }
  file_in_names.map { |name| name_queue << name }
  # Concurrent file download
  threads = $thread_count.times.map do
    Thread.new do
      Net::HTTP.start(uri.host, uri.port, :use_ssl => uri.scheme == 'https') do |http|
        while !queue.empty?
          url = queue.pop
          name = name_queue.pop unless name_queue.empty?
          if (url == nil)
            next
          end
          if(name == "")
            # Extract file name using this snippet found on SO
            begin
              name = url.meta['content-disposition'].match(/filename=(\"?)(.+)\1/)[2]
            rescue Exception => e
              name = File.basename(URI.parse(url.to_s).path)
            end
          end
          # Skip .ps files as no support for PS in textract
          if (File.extname(name) == ".ps")
            next
          end
          # Clear name of garbage and add extension
          name = name.gsub(/[\/ ]/, "").gsub(/[%20]/, "_") + ".pdf"
          if (name.size > 100)
            name = name[0..100]
          end
          if (File.exists?(name))
            print "Skip, #{name} already exists\n"
          else
            # Perform a new request for the file
            request = Net::HTTP::Get.new(url)
            request.basic_auth(ENV['IC_USERNAME'], ENV['IC_PASSWORD'])
            http.request request do |response|
              case response
              when Net::HTTPRedirection
                # Do Nothing for now
                # TODO: Fix download of secure files
              when Net::HTTPOK
                  print "Fetching #{name}\n"
                  # Write chunks of file while also reading new chunks, to increase
                  # efficiency
                  File.open(name, 'w') do |file_out|
                    response.read_body do |chunk|
                      file_out << chunk
                    end
                  end
              when Net::HTTPNotFound
                # Do nothing
              when Net::HTTPNetworkAuthenticationRequired
                puts "Wrong credentials"
              end
            end
          end
        end
      end
    end
  end
  threads.each(&:join)
end   # End of download_file_from_url

def parse(agent, args)
  $opts = []
  opt_parser = OptionParser.new do |opts|
    opts.banner = "Usage: scraper.rb [options] [path-optional]"
    opts.separator ""
    opts.separator "Specific options:"
    opts.on("-s", "--spiral", "Scrape the Imperial Spiral repository") do |a|
      $opts << "-s"
      login = agent.get("https://spiral.imperial.ac.uk/ldap-login")

      form = login.form_with(:id => 'loginform')
      username_field = form.field_with(:id => 'tlogin_netid')
      username_field.value = ENV['IC_USERNAME']
      password_field = form.field_with(:id => 'tlogin_password')
      password_field.value = ENV['IC_PASSWORD']
      button = form.button_with(:name => 'login_submit')
      loggedin_page = form.submit(button)

      spiral_scrape(agent, base_spiral_repo_url)

    end
    opts.on("-p url", "--paragraph url", "Get text from webpage paragraphs") do |p|
      $opts <<- "-p"
      src = ARGV.pop
      paragraph_scrape(agent, src)
    end
    opts.on_tail("-h", "--help", "Show this message") do
      $opts << "-h"
      puts opts
      exit
    end
  end
  opt_parser.parse!(args)
end #End parse

def main
  include Spiral
  ARGV << "-h" if ARGV.empty?
  agent = Mechanize.new
  agent.max_history=(nil)
  working_dir = Dir.pwd
  base_wikipedia_url = "https://en.wikipedia.org"
  base_spiral_repo_url = "https://spiral.imperial.ac.uk/"

  parse(agent, ARGV)

  Dir.chdir(working_dir)

end

main
