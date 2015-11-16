using DistributedArrays


addprocs(7)

nprocs()

dat=map(x->round(Int,x*10),rand(10000,10000))

ddat=distribute(dat)

map(x->myid(),dat)

map(x->myid(),ddat)

@time map(sin,dat);

@time map(sin,ddat);

function intsum(x)
  if x==0 
    0
  else
    x+intsum(x-1)
  end
end

@time reduce(+,map(intsum,dat))

@everywhere function intsum(x)
  if x==0 
    0
  else
    x+intsum(x-1)
  end
end

@time reduce(+,map(intsum,ddat))

@everywhere function rsumtail(x,n)
  if x==0
    n
  else
    rsumtail(x-1,n+x)
  end
end

@everywhere function rsumtail(x)
  rsumtail(x,0)
end

@everywhere function rsum(x)
  if x==0
    0
  else
    x+rsum(x-1)
  end
end


@time rsumtail(100000)

@time rsum(100000)

@time (s=0;
for i in dat
  s+=intsum(i)
end;
s)

@time @parallel (+) for i in dat
  intsum(i)
end

@time reduce(+,map(intsum,ddat))

ddat.indexes

ddat.chunks

fetch(@spawnat 3 sum(x->x*x,localpart(ddat)))

map(fetch,[@spawnat p sum(map(x->x*x,localpart(ddat))) for p in ddat.pids])

@time reduce(+,map(fetch,[@spawnat p sum(map(x->x*x,localpart(ddat))) for p in ddat.pids]))

@time reduce(+,map(x->x*x,ddat))
