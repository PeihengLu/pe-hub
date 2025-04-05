# compress all files larger than 50mb
find . -type f -size +50M -exec zip {}.zip {} \; 
# recreate gitignore
rm .gitignore
# add all files larger than 50mb to .gitignore without the ./ prefix but with the directory prefix
find . -type f -size +50M -exec echo {} \; | sed 's/^\.\///' >> .gitignore