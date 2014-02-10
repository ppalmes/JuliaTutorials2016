#!/usr/local/bin/julia -p 8
## syntax: ./main.jl list_of_files

require("wordcount.jl")
wordcount_files(ARGS)
