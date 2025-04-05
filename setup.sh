# deflate all zip files in their original directory
find . -type f -name "*.zip" -exec unzip -o {} \;