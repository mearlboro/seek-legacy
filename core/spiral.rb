module Spiral
  def paragraph_scrape(agent, list_url)
    # Gets page html
    page = agent.get(list_url)
    # Gets title
    title = page.parser.xpath('//h1').text
    content = ""
    content.concat(title)
    content.concat("\n")
    # Gets text inside each paragraph
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
    # Iterates the array scrapping each faculty
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
    # gets html from each faculty page
    fp = agent.get(URI.parse(URI.encode(faculty_page)));
    # gets url for new page where papers are sorted alphabetically
    search_by_title_url = fp.parser.xpath('//a[contains(@href, "/browse?type=title")]').map { |link| link['href'] }
    # gets html of alphabetical sort page
    alphabetical_search_page = agent.get(faculty_page + search_by_title_url[0])
    # scrapes the whole page
    scrape_search_page(agent, alphabetical_search_page, base_url)
    # gets url of next page and continues so recursively until no more next
    # pages are available
    next_page = alphabetical_search_page.parser.xpath('//a[contains(text(), "next")]')#.map { |link| link['href'] }
    while (next_page != nil and next_page[0] != nil)
      if (agent.head(next_page[0]['href']) != 404)
        page = agent.get(next_page[0]['href'])
        scrape_search_page(agent, page, base_url)
        next_page = page.parser.xpath('//a[contains(text(), "next")]')
      end
    end
  end

  def scrape_search_page(agent, alphabetical_search_page, base_url)

    # gets paper tags, meaning names and urls
    paper_tags = alphabetical_search_page.parser.xpath('//tr//td//a')

    paper_urls = paper_tags.map { |link| link['href'] }
    paper_names = paper_tags.map { |link| link.text.strip }

    paper_links = Array.new

    paper_urls.each do |p|
      # handles relative URIs && login
      paper_url = base_url + p
      agent.add_auth(paper_url, ENV['IC_USERNAME'], ENV['IC_PASSWORD'])
      begin
        paper_page = agent.get(paper_url)
        paper_link = paper_page.parser.xpath('//a[contains(text(), "Download")]').map { |link| link['href'] }
        # creates array of paper download links per page
        paper_links << paper_link[0]
      rescue Exception => e
        puts(e.message)
      end
    end
    thread = Thread.new do
      download_files_from_URLs(agent, Dir.pwd, paper_links, false, paper_names, URI(base_url))
    end
    thread.join
  end

end
