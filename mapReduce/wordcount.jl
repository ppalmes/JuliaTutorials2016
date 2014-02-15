# https://github.com/JuliaLang/julia/blob/master/examples/wordcount.jl
# "Map" function.
# Takes a string. Returns a HashTable with the number of times each word 
# appears in that string.
function wordcount(text)
    words=split(text,r"\s+|'\n'|'\t'|:|;|,|!|\"|\.|,",false)
    counts=Dict()
    for w in words
        counts[w]=get(counts,w,0)+1
    end
    return counts
end


# "Reduce" function.
# Takes a collection of HashTables in the format returned by wordcount()
# Returns a HashTable in which words that appear in multiple inputs
# have their totals added together.
function wcreduce(wcs)
    counts=Dict()
    for c in wcs
        for keyv in c
            (k,v)=keyv
            counts[k] = get(counts,k,0)+v
        end
    end
    return counts
end


## Splits input string into nprocs() equal-sized chunks (last one rounds up), 
## and @spawns wordcount() for each chunk to run in parallel. Then fetch()s
## results and performs wcreduce().
## Limitations: splitting the string and reduction step are single-threaded.
#function parallel_wordcount(text)
#    lines=split(text,r"'\n'",false)
#    np=nprocs()
#    unitsize=ceil(length(lines)/np)
#    wcounts={}
#    rrefs={}
#    # spawn procs
#    for i=1:np
#        first=unitsize*(i-1)+1
#        last=unitsize*i
#        if last>length(lines)
#            last=length(lines)
#        end
#        subtext=join(lines[int(first):int(last)],"\n")
#        push!(rrefs, @spawn wordcount( subtext ) )
#    end
#    # fetch results
#    while length(rrefs)>0
#        push!(wcounts,fetch(pop!(rrefs)))
#    end
#    # reduce
#    count=wcreduce(wcounts)
#    return count
#end

# Splits input string into nprocs() equal-sized chunks (last one rounds up), 
# and @spawns wordcount() for each chunk to run in parallel. Then fetch()s
# results and performs wcreduce().
# Limitations: splitting the string and reduction step are single-threaded.
function parallel_wordcount(text)
    lines=split(text,r"'\n'",false)
    np=nprocs()
    unitsize=ceil(length(lines)/np)
    wcounts={}
    rrefs={}
    # spawn procs
    res=@parallel (hcat) for i=1:np
        first=unitsize*(i-1)+1
        last=unitsize*i
        if last>length(lines)
            last=length(lines)
        end
        subtext=join(lines[int(first):int(last)],"\n")
        wordcount( subtext ) 
    end
    # reduce
    count=wcreduce([res])
    return count
end



## Takes the name of a result file, and a list of input file names.
## Combines the contents of all files, then performs a parallel_wordcount
## on the resulting string. Writes the results to result_file.
## Limitation: Performs all file IO single-threaded.
#function wordcount_files(input_file_names)
#    text=""
#    for f = input_file_names
#        fh=open(f)
#        text=join( {text,readall(fh)}, "\n" )
#        close(fh)
#    end
#    wc=parallel_wordcount(text)
#    for (k,v) = wc
#        println(k,"=",v)
#    end
#end

# Takes the name of a result file, and a list of input file names.
# Combines the contents of all files, then performs a parallel_wordcount
# on the resulting string. Writes the results to result_file.
# Limitation: Performs all file IO single-threaded.
function wordcount_files(input_file_names)
    alltext=@parallel (hcat) for f in input_file_names
        fh=open(f)
        text=readall(fh)
        close(fh)
        text
    end
    if length(input_file_names)>1
        alltext=join(alltext," ")
    end
    wc=parallel_wordcount(alltext)
    for (k,v) = wc
        println(k,"=",v)
    end
end

