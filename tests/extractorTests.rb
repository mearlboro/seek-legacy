#!/usr/bin/env ruby

require "test/unit"
require "fileutils"

class Extractor_test < Test::Unit::TestCase

    # Deletes all created files so tests can be run again
    # TODO: not make this a test
    def test_clean_up
        system('rm testExtractedData/pdf/*.txt')
    end

    def test_creates_file_in_txt_in_given_directory
        #current_directory = Dir.pwd
        
        test_file         = 'extractor_test_basics'
        src_directory     = 'testRawData/pdf/'
        dst_directory     = 'testExtractedData/pdf/'

        # Initially there is no file, and it will later on be created
        assert_equal File.file?("#{dst_directory}#{test_file}"), false

        system("python ../core/extractor.py #{src_directory}#{test_file}.pdf #{dst_directory}")

        assert_equal File.file?("#{dst_directory}#{test_file}.txt"), true
    end

    # Only tested for small files
    def test_can_extract_symbols_and_numbers
        test_file         = 'extractor_test_basics'
        src_directory     = 'testRawData/pdf/'
        dst_directory     = 'testExtractedData/pdf/'

        system("python ../core/extractor.py #{src_directory}#{test_file}.pdf #{dst_directory}")

        correct_output_file = 'testCorrectOutputData/basics.txt'
        comparison = FileUtils.compare_file("#{dst_directory}#{test_file}.txt",correct_output_file)

        assert_equal comparison,  true
    end

    def test_can_extract_bigger_text
        test_file         = 'extractor_test_small_sample'
        src_directory     = 'testRawData/pdf/'
        dst_directory     = 'testExtractedData/pdf/'

        system("python ../core/extractor.py #{src_directory}#{test_file}.pdf #{dst_directory}")

        correct_output_file = 'testCorrectOutputData/small_text.txt'
        comparison = FileUtils.compare_file("#{dst_directory}#{test_file}.txt",correct_output_file)

        assert_equal comparison,  true
    end

    def does_not_duplicate_if_file_already_exists

    end

    def can_take_single_file_as_parameter

    end

    def can_take_directory_as_parameter

    end

    def pdf_extractor_creates_file

    end

end
