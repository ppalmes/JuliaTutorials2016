#!/usr/local/bin/julia -p 8
## syntax: ./main.jl list_of_files

if length(ARGS)==0
    println("syntax: ./main.jl file1 file2 ...")
else
    print(length(ARGS))
    require("wordcount.jl")
    wordcount_files(ARGS)
end


