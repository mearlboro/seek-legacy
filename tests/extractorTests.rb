#!/usr/bin/env ruby

require 'test/unit'
require 'fileutils'
require_relative 'testUtils'

class Extractor_test < Test::Unit::TestCase

    $extractor_command = "python #{File.join('../', 'core', 'extractor.py')}"

    # Test can work with .docx as well. Used pdf just to
    # test wether it CREATES the file or not
    def test_creates_file_in_txt_in_given_directory
        test_file         = 'extractor_test_basics'
        src_directory     = File.join('testRawData', 'pdf/')
        dst_directory     = File.join('testExtractedData', 'pdf/')

        # Initially there is no file, and it will later on be created
        assert_equal File.file?("#{dst_directory}#{test_file}"), false

        system("#{$extractor_command} #{src_directory}#{test_file}.pdf #{dst_directory}")

        assert_equal File.file?("#{dst_directory}#{test_file}.txt"), true
    end

    # Only tested for small files
    def test_can_extract_symbols_and_numbers
        test_file         = 'extractor_test_basics'
        src_directory     = File.join('testRawData', 'pdf/')
        dst_directory     = File.join('testExtractedData', 'pdf/')

        system("#{$extractor_command} #{src_directory}#{test_file}.pdf #{dst_directory}")

        correct_output_file = File.join('testCorrectOutputData','basics.txt')
        comparison = FileUtils.compare_file("#{dst_directory}#{test_file}.txt",correct_output_file)

        assert_equal comparison,  true
    end

    def test_can_extract_bigger_text
        test_file         = 'extractor_test_small_sample'
        src_directory     = File.join('testRawData', 'pdf/')
        dst_directory     = File.join('testExtractedData', 'pdf/')

        system("#{$extractor_command} #{src_directory}#{test_file}.pdf #{dst_directory}")

        correct_output_file = File.join('testCorrectOutputData', 'small_text.txt')
        comparison = FileUtils.compare_file("#{dst_directory}#{test_file}.txt",correct_output_file)

        assert_equal comparison,  true
    end


    # Test the non-duplication of files by counting the number of files in directory
    def test_does_not_duplicate_if_file_already_exists
        test_file         = 'extractor_test_small_sample'
        src_directory     = File.join('testRawData', 'pdf/')
        dst_directory     = File.join('testExtractedData', 'pdf/')

	number_files_before = TestUtils.count_files_directory(dst_directory)

        # Order of tests is not guaranteed, so create the file if necessary
        if (not File.exists?("#{dst_directory}#{test_file}.txt"))
            system("#{$extractor_command} #{src_directory}#{test_file}.pdf #{dst_directory}")
            number_files_before = number_files_before + 1
        end
        system("#{$extractor_command} #{src_directory}#{test_file}.pdf #{dst_directory}")
        number_of_files_after = TestUtils.count_files_directory(dst_directory)
        
        assert_equal number_files_before, number_of_files_after
    end

end

# Deletes all extracted files before running tests
puts 'Deleting previous test results'
system("rm testExtractedData/pdf/*.txt")
