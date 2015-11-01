#!/usr/bin/env ruby

require 'open-uri'
require 'mechanize'
require 'nokogiri'
require 'net/http'

def main
  agent = Mechanize.new
  base_wikipedia_url = "https://en.wikipedia.org"
  list_url = "#{base_wikipedia_url}/wiki/Marie_Curie";
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

main
