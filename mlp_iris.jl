# Author: Paulito Palmes
# Parallel cross-validation of Iris dataset using Knet Neural Network Package
# Date: September 2, 2016

nprocs() == 1 && addprocs()
using RDatasets
using Knet
using MLBase

@everywhere using Knet
@everywhere using RDatasets
@everywhere using MLBase


@everywhere function toBit{T<:Integer}(x::T,nclass::T)
    assert(x>0 && x<=nclass)
    v = zeros(nclass)
    v[x] = 1
    return v
end

@everywhere function getData()
    iris = dataset("datasets","iris")
    ndx = randperm(nrow(iris))
    iris = iris[ndx,:]
    x = Matrix(iris[:,1:4])'
    label = Array(iris[:,5])
    lm = labelmap(label)
    y = labelencode(lm,label)
    label=@parallel (hcat) for i in y
        toBit(i,3)
    end
    #normalize features
    xnorm = (x .- mean(x,2)) ./ std(x,2)
    #split training/testing
    xtrain = xnorm[:,1:75]
    ytrain = label[:,1:75]
    xtest = xnorm[:,76:end]
    ytest = label[:,76:end]
    return (xtrain,ytrain,xtest,ytest)
end

@everywhere function train(f, data)
    for (x,y) in data
        forw(f, x)
        back(f, y, softloss)
        update!(f)
    end
end

@everywhere function test(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    return sumloss / numloss
end


#create mlp
@everywhere Knet.@knet function softmax(x)
    w1 = par(init=Gaussian(0,0.001), dims=(20,4))
    b1 = par(init=Constant(0), dims=(20,1))
    y1 = relu(w1 * x .+ b1)
    w2 = par(init=Gaussian(0,0.001), dims=(3,20))
    b2 = par(init=Constant(0), dims=(3,1))
    return soft(w2 * y1 .+ b2)
end

@knet function softmax(x)
    w1 = par(init=Gaussian(0,0.001), dims=(30,4))
    b1 = par(init=Constant(0), dims=(30,1))
    y1 = (w1 * x .+ b1)
    return (y1)
end

@everywhere function crossvalIris(lRate,epochs)
    #epochs = 200 
    ksoftmax = Knet.compile(:softmax);
    setp(ksoftmax;lr=lRate);
    (xtrain,ytrain,xtest,ytest) = getData()
    data = minibatch(xtrain,ytrain,5)
    tdata = minibatch(xtest,ytest,5)
    for i=1:epochs
        train(ksoftmax,data)
        #println(test(ksoftmax,tdata,zeroone))
    end
    pred = forw(ksoftmax,xtest);
    println(confusmat(3,classify(pred),classify(ytest)))
    1-errorrate(classify(pred),classify(ytest))
end

fTable = @parallel (vcat) for epochs in [100,200,300,500]
    lrTable=@parallel (vcat) for lRate in [0.001,0.01,0.015,0.1,0.15,5]
        res=@parallel (vcat) for i=1:10
            crossvalIris(lRate,epochs)
        end
        println((lRate,mean(res),std(res),length(res)))
        [epochs lRate mean(res) std(res) length(res)]
    end
end

sorted = sortrows(fTable,by=x->x[3]);
sorted = DataFrame(sorted);
rename!(sorted,Dict(:x1=>:EPOCHS,:x2=>:LR,:x3=>:ACC,:x4=>:SD,:x5=>:N))

