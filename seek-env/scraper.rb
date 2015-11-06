#!/usr/bin/env ruby
begin
  gem 'mechanize', ">=2.7"
rescue Gem::LoadError => e
  system("sudo gem install mechanize")
  Gem.clear_paths
end

require 'open-uri'
require 'mechanize'
require 'nokogiri'
require 'net/http'
require 'io/console'

def download_files_from_URLs(agent, target_dir, links, override, file_in_names, thread_count, current_link)
  # Open file from web URL, using username and password provided
  include FormatMethods

  uri = URI(current_link)

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
          if(name == "")
            # Extract file name using this snippet found on SO
            begin
              name = url.meta['content-disposition'].match(/filename=(\"?)(.+)\1/)[2]
            rescue Exception => e
              name = File.basename(URI.parse(url.to_s).path)
            end
          end
          if (File.extname(name) == ".ps")
            next
          end
          name = name.gsub(/[\% ]/, "")

          if (File.exists?(name))
            print "Skip, #{name} already exists\n"
          else
            request = Net::HTTP::Get.new(url)
            request.basic_auth(ENV['IC_USERNAME'], ENV['IC_PASSWORD'])
            http.request request do |response|
              case response
              when Net::HTTPRedirection
                # Do Nothing for now TODO: Fix download of secure files
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
  threads = Array.new(4)
  faculty_pages.each do |faculty_page|
    threads << Thread.new do
      scrape_faculty(agent, faculty_page, base_url)
    end
  end
  threads.each do |t|
    if !t.nil?
      t.join
    end
  end
end

def scrape_faculty(agent, faculty_page, base_url)
  fp = agent.get(URI.parse(URI.encode(faculty_page)));
  search_by_title_url = fp.parser.xpath('//a[contains(@href, "/browse?type=title")]').map { |link| link['href'] }
  alphabetical_search_page = agent.get(faculty_page + search_by_title_url[0])
  scrape_search_page(agent, alphabetical_search_page, base_url)
  next_page = alphabetical_search_page.parser.xpath('//a[contains(text(), "next")]')#.map { |link| link['href'] }
  while (next_page != nil)
    page = agent.get(next_page[0]['href'])
    scrape_search_page(agent, page, base_url)
    next_page = page.parser.xpath('//a[contains(text(), "next")]')#.map { |link| link['href'] }
  end
end

def scrape_search_page(agent, alphabetical_search_page, base_url)
  paper_tags = alphabetical_search_page.parser.xpath('//tr//td//a')

  paper_urls = paper_tags.map { |link| link['href'] }
  # paper_names = paper_tags.map { |link| link.text.strip }
  paper_names = Array.new(paper_urls.length).fill("")

  paper_links = Array.new

  paper_urls.each do |p|
    paper_url = base_url + p
    agent.add_auth(paper_url, ENV['IC_USERNAME'], ENV['IC_PASSWORD'])
    paper_page = agent.get(paper_url)
    paper_link = paper_page.parser.xpath('//a[contains(text(), "Download")]').map { |link| link['href'] }
    paper_links << paper_link[0]
  end
  thread = Thread.new do
    download_files_from_URLs(agent, Dir.pwd, paper_links, false, paper_names, 10, URI(base_url))
  end
  thread.join
end

def main
  agent = Mechanize.new
  working_dir = Dir.pwd
  Dir.mkdir("../PDF") unless File.exists?("../PDF")
  Dir.chdir("../PDF")
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
  Dir.chdir(working_dir)

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
