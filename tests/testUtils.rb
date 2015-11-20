# Module used for extracting the non-test related behaviour within 
# the testing enviroment

module TestUtils

    def self.count_files_directory(dir)
        Dir[File.join(dir, '**', '*')].count { |file| File.file?(file) }
    end
end
