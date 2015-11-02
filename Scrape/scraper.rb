#!/usr/bin/env ruby

require 'open-uri'
require 'mechanize'
require 'nokogiri'
require 'net/http'
require 'io/console'

def download_files_from_URLs(target_dir, links, override, file_in_names, thread_count, current_link)
  # Open file from web URL, using username and password provided
  include FormatMethods

  uri = URI(current_link)
  working_dir = Dir.pwd
  Dir.mkdir(target_dir) unless File.exists?(target_dir)
  # create_directory(target_dir)
  Dir.chdir(target_dir)

  queue = Queue.new
  name_queue = Queue.new

  links.map { |url| queue << url }
  file_in_names.map { |name| name_queue << name }

  threads = thread_count.times.map do
    Thread.new do
      Net::HTTP.start(uri.host, uri.port, :use_ssl => uri.scheme == 'https') do |http|
        while !queue.empty?
          url = queue.pop
          name = name_queue.pop unless name_queue.empty?
          if (url == nil)
            next
          end
          if(name == "" || name == "Slides")
            # Extract file name using this snippet found on SO
            begin
              name = url.meta['content-disposition'].match(/filename=(\"?)(.+)\1/)[2]
            rescue Exception => e
              # puts "Unable to find file name" + e.message
              name = File.basename(URI.parse(url.to_s).path)
            end
          end
          if (File.extname(name) == "")
            name += ".pdf"
          end
          if (File.exists?(name))
            print "Skip, #{name} already exists\n"
          else

            # request = Net::HTTP::Post.new(uri + '/ldap-login')
            # request.basic_auth(ENV['IC_USERNAME'], ENV['IC_PASSWORD'])
            # request.set_form_data(:tlogin_netid => ENV['IC_USERNAME'], :tlogin_password => ENV['IC_PASSWORD'])
            # response = http.request request

            request = Net::HTTP::Get.new(url)
            http.request request do |response|
              case response
              when Net::HTTPRedirection
                req = Net::HTTP::Post.new(response['location'])
                req.set_form_data(:tlogin_netid => ENV['IC_USERNAME'], :tlogin_password => ENV['IC_PASSWORD'])
                http.request req do |res|
                  case res
                  when Net::HTTPOK
                    print "Fetching #{name}\n"
                    File.open(name, 'w') do |file_out|
                      res.read_body do |chunk|
                        file_out << chunk
                      end
                    end
                  end
                end
              when Net::HTTPOK
                  print "Fetching #{name}\n"
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
  Dir.chdir(working_dir)
end   # End of download_file_from_url

def wikipedia_scrape(agent, list_url)
  page = agent.get(list_url)
  title = page.parser.xpath('//h1').text
  content = ""
  content.concat(title)
  content.concat("\n")
  page.parser.xpath('//p').each do |info|
    content.concat(info.text)
    content.concat("\n")
  end
  File.open(title + ".txt", 'w') do |f|
    f << content
  end
end

def spiral_scrape(agent, base_url)
  main_page = agent.get(base_url)
  engineering_faculty = main_page.parser.xpath('//div[@class="tp_download1 col-md-6 col-lg-3"]/a').map { |link| link['href'] }
  medicine_faculty = main_page.parser.xpath('//div[@class="tp_download2 col-md-6 col-lg-3"]/a').map { |link| link['href'] }
  ns_faculty = main_page.parser.xpath('//div[@class="tp_download3 col-md-6 col-lg-3"]/a').map { |link| link['href'] }
  business_faculty = main_page.parser.xpath('//div[@class="tp_download4 col-md-6 col-lg-3"]/a').map { |link| link['href'] }
  faculty_pages = Array.new
  faculty_pages << engineering_faculty[0]
  faculty_pages << medicine_faculty[0]
  faculty_pages << ns_faculty[0]
  faculty_pages << business_faculty[0]

  engineering_page = agent.get(URI.parse(URI.encode(faculty_pages[0])));
  search_by_title = engineering_page.parser.xpath('//a[contains(@href, "/browse?type=title")]').map { |link| link['href'] }
  alphabetical_search_page = agent.get(faculty_pages[0] + search_by_title[0])

  paper_urls = alphabetical_search_page.parser.xpath('//tr//td//a')

  paper_pages = paper_urls.map { |link| link['href'] }
  paper_names = paper_urls.map { |link| link.text.strip }
  paper_links = Array.new

  paper_pages.each do |p|
    paper_url = base_url.concat(p)
    agent.add_auth(paper_url, ENV['IC_USERNAME'], ENV['IC_PASSWORD'])
    paper_page = agent.get(paper_url)
    paper_link = paper_page.parser.xpath('//a[contains(text(), "Download")]').map { |link| link['href'] }
    paper_links << paper_link[0]
  end
  # puts paper_names
  # puts
  download_files_from_URLs("PDF", paper_links, false, paper_names, 4, alphabetical_search_page.uri)
  # puts alphabetical_search.uri.to_s
end

def main
  agent = Mechanize.new
  base_wikipedia_url = "https://en.wikipedia.org"
  base_spiral_repo_url = "https://spiral.imperial.ac.uk/"
  list_url = "#{base_wikipedia_url}/wiki/Marie_Curie";
  login = agent.get("https://spiral.imperial.ac.uk/ldap-login")

  form = login.form_with(:id => 'loginform')
  username_field = form.field_with(:id => 'tlogin_netid')
  username_field.value = ENV['IC_USERNAME']
  password_field = form.field_with(:id => 'tlogin_password')
  password_field.value = ENV['IC_PASSWORD']
  button = form.button_with(:name => 'login_submit')

  loggedin_page = form.submit(button)
  spiral_scrape(agent, base_spiral_repo_url)
  # wikipedia_scrape(agent, list_url)
end

module FormatMethods
  @@rows, @@cols = IO.console.winsize
  def print_equal
    for i in 1..@@cols
      print "="
    end
  end # End print_equal

  def print_loading
    print "["
    for i in 2..@@cols-2
      sleep(1.0/60.0)
      print "#"
    end
    print "]"
  end
  def format_text(text)
    return text.gsub("\t\n ", "").strip
  end
end


main
