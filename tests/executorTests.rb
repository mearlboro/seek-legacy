#!/usr/bin/env ruby

require 'test/unit'
require_relative 'testUtils'

class Executor_test < Test::Unit::TestCase

    $executor_command   = "ruby executor.rb"
    $dst_directory      = File.join('..', 'tests' ,'testExtractedData', 'pdf/')
    #Because the executor calls extractor.py, we need to change directory
    #at the beginning of each test
    $executor_directory = File.join('..', 'core')

    def test_can_take_single_file_as_parameter
        Dir.chdir $executor_directory
        test_file         = 'extractor_test_basics'
        src_directory     = File.join('..', 'tests', 'testRawData', 'pdf/')

        system("#{$executor_command}  #{src_directory}#{test_file}.pdf #{$dst_directory}")
        assert(File.exists?("#{$dst_directory}extractor_test_basics.txt") , 'Files were not generated')

        clean_directory
    end

    def test_can_take_directory_as_parameter
        Dir.chdir $executor_directory
        src_directory = File.join('..', 'tests', 'testRawData', 'testSingleFolder/')
        before, after, total  = extract_files_and_return_count(src_directory)
       
        assert(before < after, 'Files were not generated')
        assert(total == after, 'not enough files were generated')

        clean_directory
    end

    def test_does_extract_files_on_subfolders
        Dir.chdir $executor_directory
        src_directory = File.join('..', 'tests', 'testRawData', 'testSubFolders/') 
        before, after, total  = extract_files_and_return_count(src_directory)

        puts "before: #{before} after: #{after}"
        assert(before < after, 'Files were not generated')
        assert(total == after, 'not enough files were generated')

        clean_directory 
    end

    # Runs the executor and returns the values to further check wether the
    # right amount of files have been created
    def extract_files_and_return_count(src_directory)
        Dir.chdir $executor_directory

        
        number_files_before = TestUtils.count_files_directory($dst_directory)
        number_files_src_directory = TestUtils.count_files_directory(src_directory)

        system("#{$executor_command}  #{src_directory} #{$dst_directory}")

        number_files_after = TestUtils.count_files_directory($dst_directory)

        total_correct_amount_of_files = number_files_before + number_files_src_directory
        return number_files_before, number_files_after, total_correct_amount_of_files

	clean_directory
    end

    def clean_directory
        Dir.chdir '../tests'
        system("rm #{$dst_directory}*.txt")
    end
end

# Deletes all extracted files before running tests
puts 'Deleting previous test results'

system("rm testExtractedData/pdf/*.txt")
