#NOTE:
#
#Each process has an associated identifier. The process providing the interactive julia prompt always has an id equal to 1, as would the julia process running the driver script in the example above. The processors used by default for parallel operations are referred to as workers. When there is only one process, process 1 is considered a worker. Otherwise, workers are considered to be all processes other than process 1.
#
#The base Julia installation has in-built support for two types of clusters:
#
#        1. A local cluster specified with the -p option as shown above.
#        2. And a cluster spanning machines using the --machinefile option. This uses ssh to start the worker processes on the specified machines.
#
#Functions addprocs, rmprocs, workers and others, are available as a programmatic means of adding, removing and querying the processes in a cluster.

workers()


procs()


addprocs(7)

r = remotecall(3, rand, 2, 2)

b=fetch(r)

s = @spawnat 2 rand(2,2)

fetch(s)

remotecall_fetch(2, rand, 2,2)

remotecall_fetch(2, getindex,r,1,1)

r = @spawn rand(2,2)

fetch(r)

s = @spawn 1+fetch(r)

fetch(s)

a=RemoteRef[]
for i in 1:10
    push!(a,@spawn sin(i))
end

@time sum(map(fetch,a))

@time @parallel (+) for i in a
  fetch(i)
end

@time sum(map((x)->fetch(x),a))

@everywhere function producer()
  for i in 1:3
    produce(i)
  end
end
task=Task(producer)
consume(task)

consume(task)

n=10;
@everywhere function ssin() 
    for i in 1:n
        produce(sin(i))
    end
end
p=Task(ssin)
@time reduce(+,map(x->consume(p),1:n))

s=0
@time for i in 1:10000000
  s+=sin(i)
end
s

@time @parallel (+) for i=1:10000000
  sin(i)
end

@time reduce(+,(map(i->sin(i),1:10000000)))

@time reduce(+,map(sin, 1:10000000))

@time sum( map(1:10000000) do x
         sin(x);
     end
)

# method 1
A = rand(1000,1000)
Bref = @spawn A^2
...
fetch(Bref)

# method 2
Bref = @spawn rand(1000,1000)^2
...
fetch(Bref)

@everywhere function count_heads(n)
    c::Int = 0
    for i=1:n
        c += rand(Bool)
    end
    c
end


@time count_heads(200000000)/200000000

#require("count_heads")

@time begin
a = @spawn count_heads(100000000)
b = @spawn count_heads(100000000)
(fetch(a)+fetch(b))/200000000
end

@time begin
nheads = @parallel (+) for i=1:200000000
    round(Int,rand(Bool))
end
nheads/200000000
end

@time begin
    a=0
    for i=1:20000000
      a+=round(Int,rand(Bool))
    end
    a/20000000
end

@time sum(map(x->round(Int,rand(Bool)),1:2000000)/2000000

b = zeros(100000);

c=@parallel (hcat) for i=1:10
    i
end

print(c);


