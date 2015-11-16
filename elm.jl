include("MyELM.jl")

import RDatasets
import DataFrames


##-------------------------------
## iris classification
## Test classify function
#
#function classify(str::String) 
#    if str == "setosa"
#        return 0 
#    elseif str == "versicolor"
#        return 1 
#    elseif str == "virginica"
#        return 2 
#    end
#    return 4
#end
#
#iris = RDatasets.dataset("datasets","iris")
#r,c=size(iris)
#class=zeros(Int64,r)
#for i in 1:r
#    class[i] = classify(iris[i,5])
#end
#
#ndx = randperm(r)
#trndx = ndx[1:75]
#tstndx = ndx[76:150]
#trX = array(iris[trndx,1:4])
#tstX = array(iris[tstndx,1:4])
#trY = class[trndx]
#tstY = class[tstndx]
#MyELM.classify([10,10],trX,trY,tstX,tstY)

#----------------------------------------

##-----------------------------------------
## Test curvefit function for regression
#
#airquality = readtable("air.txt")
#r,c = size(airquality)
#ndx = randperm(r)
#trndx = ndx[1:int(r/2)]
#tstndx = ndx[int(r/2)+1:end]
#trX = array(airquality[trndx,[1,2,4,5,6]])
#tstX = array(airquality[tstndx,[1,2,4,5,6]])
#trY = vec(array(airquality[trndx,3]))
#tstY = vec(array(airquality[tstndx,3]))
#MyELM2.curvefit(800,trX,trY,tstX,tstY);

#
#-----------------------------------------

#--------------------------------
#trX = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 7.0]
#trY = vec([3.0,7.0,11.0,14.0])
#tstX = [1.0 2.0; 3.0 3.0]
#tstY = [3.0; 6.0]
#MyELM.curvefit([10,10],trX,trY,tstX,tstY)

#model = ELM(2,[10,10],1,sigmoid)
#train(model,x,y)
#println(predictVal(model,x))
#println(predictVal(model,x1))
#-------------------------------

#-------------------------------
#myelm = ELM(2,3,2,sigmoid)
#x = [0.0 1.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
#y = vec([1,1,1,0])
#train(myelm,x,y)
#predict(myelm,x)
#-------------------------------

#-------------------------------
dat_train = readdlm("mnist_train.csv",',',Float64)
dat_test = readdlm("mnist_test.csv",',',Float64)

trainX=dat_train[:,2:end]
trainY=int(dat_train[:,1])

testX=dat_test[:,2:end]
testY=int(dat_test[:,1])

@time model = MyELM.classify([30,500],trainX,trainY,testX,testY)

##mnist = MyELM(size(trainX)[2],200,10,sigmoid)
##@time train(mnist,trainX,trainY)
##res = predict(mnist,testX)
##println(mean(res .== testY))

#f = open("myelm.serialized","w")
#serialize(f,myelm)
#close(f)
#f = open("myelm.serialized","r")
#m = deserialize(f)
#-----------------------------------
