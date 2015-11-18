#!/usr/bin/env ruby

require 'test/unit'
require_relative 'testUtils'

class Executor_test < Test::Unit::TestCase

    $executor_command = "ruby #{File.join('../', 'core', 'executor.rb')}"
    $dst_directory    = File.join('testExtractedData', 'pdf/')

    def test_can_take_single_file_as_parameter
        test_file         = 'extractor_test_basics'
        src_directory     = File.join('testRawData', 'pdf/')

        system("#{$executor_command}  #{src_directory}#{test_file}.pdf #{$dst_directory}")
        assert(File.exists?("#{$dst_directory}#{test_file}.txt") , 'Files were not generated')

        clean_directory
    end

    def test_can_take_directory_as_parameter
        src_directory = File.join('testRawData', 'testSingleFolder/')
        after, total  = extract_files_and_return_count(src_directory)

        assert(total == after, 'Files were not generated')

        clean_directory
    end

    def test_does_extract_files_on_subfolders
        src_directory = File.join('testRawData', 'testSubFolders/') 
        after, total  = extract_files_and_return_count(src_directory)

        assert(total == after, 'Files were not generated')

        clean_directory 
    end

    # Runs the executor and returns the values to further check wether the
    # right amount of files have been created
    def extract_files_and_return_count(src_directory)
        number_files_before = TestUtils.count_files_directory($dst_directory)
        number_files_src_directory = TestUtils.count_files_directory(src_directory)

        system("#{$executor_command}  #{src_directory} #{$dst_directory}")

        number_files_after = TestUtils.count_files_directory($dst_directory)

        total_correct_amount_of_files = number_files_before + number_files_src_directory
        return number_files_after, total_correct_amount_of_files
    end

    def clean_directory
        system("#{$dst_directory}*.txt")
    end
end

# Deletes all extracted files before running tests
puts 'Deleting previous test results'
system("rm testExtractedData/pdf/*.txt")
