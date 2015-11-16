#!/usr/local/bin/julia -p 10

f=function(n) 
    return(randn(n,n))
end

if length(ARGS)>0
    n=int(ARGS[1])
else
    n=100
end

println(f(n))
