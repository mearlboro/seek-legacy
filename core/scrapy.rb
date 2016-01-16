module Scrapy

  # Seek: scrapy.rb
  # Scraper module, allows for easy customization

  # This module contains all the different capabilities currently available for
  # scraping via Seek.
  # Each method performs a different task which will be described before each
  # definition

  # Attempts scraping of a whole Wikipedia category
  # Has been tested on people of the 20th Century in Europe
  # The sleep is there so that the Wikipedia so that the robots.txt file does
  # not ban the scrapper.
  def wiki_scrape(agent, url)
    # TODO: Update path
    Dir.chdir("../raw/text/")

    page = agent.get(url)

    links = page.parser.xpath('//a[contains(@class, "CategoryTreeLabel")]')
    links.each do |link|
      citizenship_page = agent.get(link['href'])

      Dir.mkdir(link.text()) unless File.exists?(link.text())
      Dir.chdir(link.text())

      sleep(1)
      # Selects all URLs from the list existing inside a div with the class
      # mw-category-group
      people = citizenship_page.parser.xpath('//div[@class="mw-category-group"]/ul/li/a').map { |link| link['href'] }
      sleep(0.5)

      people.each do |person|
        paragraph_scrape(agent, person)
        sleep(0.5)
      end

      Dir.chdir("..")
    end
  end

  # Parses a text file for collocations, creating the tuples as it reads through
  def get_tuple(agent, src)
    solution = File.open("answer.txt", "w")
    html_dir = File.dirname(__FILE__)

    page = agent.get("file:///#{html_dir}/#{src}")
    rows = page.parser.xpath('//tr')

    ln = rows.length - 1
    rows[2..ln].each do |row|
      columns = Nokogiri::HTML(row.inner_html).xpath('//td')
      if (columns.length > 1)
        solution << columns[0].text() + ": " + columns[4].text()
      end
    end
  end

  # Scrapes a page for all the text inside <p> tags
  def paragraph_scrape(agent, list_url)
    page = agent.get(list_url)

    title = page.parser.xpath('//h1').text.gsub(/[ ]/, "_")

    content = ""
    content << title
    content << "\n"

    # Retrieves text inside <p> tag
    page.parser.xpath('//p').each do |info|
      content << info.text
      content << "\n"
    end

    File.open(title + ".txt", 'w') do |f|
      f << content
    end
  end

  # Scrapes the Imperial College London Spiral repository for all papers available
  def spiral_scrape(agent, base_url)
    main_page = agent.get(base_url)

    # Gets each faculty search page url
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
    # Scrape each faculty on a new thread
    faculty_pages.each do |faculty_page|
      threads << Thread.new do
        scrape_faculty(agent, faculty_page, base_url)
      end
    end

    # Ensure threads don't kill each other
    threads.each do |t|
      if !t.nil?
        t.join
      end
    end
  end

  # Helper method: Scrapes each faculty page
  def scrape_faculty(agent, faculty_page, base_url)
    # Gets HTML from each faculty page
    fp = agent.get(URI.parse(URI.encode(faculty_page)));
    # Gets URL for new page where papers are sorted alphabetically
    search_by_title_url = fp.parser.xpath('//a[contains(@href, "/browse?type=title")]').map { |link| link['href'] }
    # Retrieves HTML of alphabetical sort page
    alphabetical_search_page = agent.get(faculty_page + search_by_title_url[0])
    # Scrapes the whole page
    scrape_search_page(agent, alphabetical_search_page, base_url)
    # Retrieves URL of next page and continues so recursively until no more
    # pages are available
    next_page = alphabetical_search_page.parser.xpath('//a[contains(text(), "next")]')
    # Recursively scrape each page for a faculty
    while (next_page != nil and next_page[0] != nil)
      if (agent.head(next_page[0]['href']) != 404)
        page = agent.get(next_page[0]['href'])
        scrape_search_page(agent, page, base_url)
        next_page = page.parser.xpath('//a[contains(text(), "next")]')
      end
    end
  end

  # Helper method: Scrapes the original search page
  def scrape_search_page(agent, alphabetical_search_page, base_url)

    # Retrieves paper tags, meaning names and URLs
    paper_tags = alphabetical_search_page.parser.xpath('//tr//td//a')
    paper_urls = paper_tags.map { |link| link['href'] }
    paper_names = paper_tags.map { |link| link.text.strip }

    # Creates array of paper download links per page
    paper_links = Array.new
    paper_urls.each do |paper|
      # handles relative URIs && login
      paper_url = base_url + paper
      agent.add_auth(paper_url, ENV['IC_USERNAME'], ENV['IC_PASSWORD'])
      begin
        paper_page = agent.get(paper_url)
        paper_link = paper_page.parser.xpath('//a[contains(text(), "Download")]').map { |link| link['href'] }
        paper_links << paper_link[0]
      rescue Exception => e
        puts(e.message)
      end
    end

    thread = Thread.new do
      download_files_from_URLs(agent, Dir.pwd, paper_links, paper_names, URI(base_url))
    end
    thread.join
  end
end
