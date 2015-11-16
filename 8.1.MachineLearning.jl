
if nprocs()==1
    addprocs(7)
end

@everywhere using RDatasets
@everywhere using DataFrames
@everywhere using Clustering
@everywhere using LIBSVM
@everywhere using Gadfly
@everywhere using DecisionTree
@everywhere using DimensionalityReduction

# read data
iris = RDatasets.dataset("datasets","iris");

head(iris)

features=array(iris[:,[1:4]])
Gadfly.set_default_plot_size(20cm, 10cm)
nc=3
km=Clustering.kmeans(features',nc);
centers=km.centers'

feature1=1
feature2=2
l1=layer(x=iris[:,feature1],y=iris[:,feature2],Geom.point,color=iris[:,5]);
l2=layer(x=centers[:,feature1],y=centers[:,feature2],Geom.point,Theme(default_color=color("black"),default_point_size=12mm));
plot(l1,l2)

using DimensionalityReduction

#data=array(iris[:,1:4])
data=convert(Array,DataArray(iris[:,1:4]))
pcaIris=DimensionalityReduction.pca(data);
scores=pcaIris.scores
#plot(x=scores[:,1],y=scores[:,2],Geom.point,color=iris[:Species])
nc=3
km=Clustering.kmeans(scores',nc);
centers=km.centers'
feature1=1
feature2=2
l1=layer(x=scores[:,feature1],y=scores[:,feature2],Geom.point,color=iris[:,5]);
l2=layer(x=centers[:,feature1],y=centers[:,feature2],Geom.point,Theme(default_color=color("black"),default_point_size=10mm));
plot(l1,l2)

@everywhere using RDatasets
@everywhere using LIBSVM
@everywhere using DecisionTree
iris = RDatasets.dataset("datasets","iris");

features = array(iris[:,1:4])
labels = array(iris[:,5])
sz,ft = size(features)
train = randbool(sz)
test = !train;

v=@parallel (vcat) for i=1:1000
    train = randbool(sz)
    test = !train;
    trInp = features[train,2:4]
    trObs = labels[train]
    tstInp = features[test,2:4]
    tstObs = labels[test]
    model = svmtrain(trObs, trInp'); 
    (predicted_labels, decision_values)=svmpredict(model, tstInp')
    res=DecisionTree.confusion_matrix(predicted_labels,tstObs)
    res.accuracy
end
(mean(v),std(v))

features = scores;

v = @parallel (hcat) for i in 1:1000
    train = randbool(sz)
    test = !train;
    trInp = features[train,:]
    trObs = labels[train]
    tstInp = features[test,:]
    tstObs = labels[test]
    model = svmtrain(trObs, trInp'); 
    (predicted_labels, decision_values)=svmpredict(model, tstInp')
    dec=DecisionTree.confusion_matrix(predicted_labels,tstObs)
    dec.accuracy
end
(mean(v),std(v))

@everywhere using DecisionTree

features = array(iris[:,1:4]);
labels = array(iris[:,5]);
sz,ft = size(features);


params=@parallel (vcat) for ntree=10:10:100
    v=@parallel (hcat) for i in 1:100
        train = randbool(sz)
        test = !train;
        trInp = features[train,:]
        trObs = labels[train]
        tstInp = features[test,:]
        tstObs = labels[test]
        modelrf=DecisionTree.build_forest(trObs,trInp,4,ntree);
        predrf=DecisionTree.apply_forest(modelrf,tstInp);
        dec=DecisionTree.confusion_matrix(tstObs,predrf);
        dec.accuracy
    end
    [ntree mean(v) std(v)]
end
#params=convert(DataFrame,params)
params=convert(DataFrame,params);
ymn=params[:,2]-params[:,3]
ymx=params[:,2]+params[:,3]
plot(params,x=:x1,y=:x2,ymin=ymn,ymax=ymx,Geom.line,Scale.y_continuous(minvalue=0.7, maxvalue=1),Geom.point,Geom.errorbar)

params

using DecisionTree
params=Array(Float64,0,3)
for ntree=5:5:50
    v = @parallel (vcat) for i in 1:10
            features = array(iris[:,1:4]);
            labels = array(iris[:,5]);
            sz,ft = size(features);
            train = randbool(sz)
            test = !train;
            trInp = features[train,:]
            trObs = labels[train]
            tstInp = features[test,:]
            tstObs = labels[test]
            modelad,coeffs=DecisionTree.build_adaboost_stumps(trObs,trInp,ntree);
            predad=DecisionTree.apply_forest(modelad,tstInp);
            dec=DecisionTree.confusion_matrix(tstObs,predad);
            dec.accuracy
    end
    params=vcat(params,[ntree mean(v) std(v)])
end
params=convert(DataFrame,params)
params=convert(DataFrame,params);
ymn=params[:,2]-params[:,3]
ymx=params[:,2]+params[:,3]
plot(params,x=:x1,y=:x2,ymin=ymn,ymax=ymx,Geom.line,Scale.y_continuous(minvalue=0.0, maxvalue=1),Geom.point,Geom.errorbar)

params

@everywhere using RDatasets

air=RDatasets.dataset("datasets","airquality")
head(air)

airc=air[complete_cases(air),:];
head(airc)

cor(array(airc))

plot(airc,x=:Day,y=:Temp,Geom.point,Geom.smooth)

using RDatasets
using DataFrames
using Clustering
using LIBSVM
using DecisionTree
using DimensionalityReduction
using Gadfly

rmse(x,y) = sqrt(mean((x-y).^2))

air=RDatasets.dataset("datasets","airquality")
airc=air[complete_cases(air),:];
sz,ft = size(airc);
train = randbool(sz)
test = !train

airc = convert(Array{Float64,2},DataArray(airc))

ftcols=[1,2,3,5,6]
labcol = 4

features= airc[:,ftcols]
labels = airc[:,labcol]

trInp = features[train,:]
trObs= labels[train]
tstInp = features[test,:]
tstObs = labels[test]

# modelsvm = svmtrain(trObs, trInp');
# (predicted_output, decision_values)=svmpredict(modelsvm, tstInp')

modelrf = DecisionTree.build_forest(trObs,trInp,2,100);
predrf=DecisionTree.apply_forest(modelrf,tstInp);

print("RMSE: ",rmse(tstObs,predrf))

data=convert(DataFrame,[predrf tstObs])
xs = 1:nrow(data)

Gadfly.set_default_plot_size(20cm, 10cm)
p1=plot(layer(data,x=xs,y=:x1,Geom.line),layer(data,x=xs,y=:x2,Geom.line,Theme(default_color=color("black"))))
p2=plot(layer(x=xs,y=data[:x1]-data[:x2],Geom.point),layer(x=0:nrow(data),y=rep(0,nrow(data)),Geom.line))
p3=plot(layer(x=data[:x1]-data[:x2],Geom.density))
hstack(p1,p2,p3)

using RDatasets
using DataFrames
using Clustering
using LIBSVM
using DecisionTree
using DimensionalityReduction
using Gadfly

rmse(x,y) = sqrt(mean((x-y).^2))

air=RDatasets.dataset("datasets","airquality")

airc=air[complete_cases(air),:];



sz,ft = size(airc);
train = randbool(sz)
test = !train

airc = convert(Array{Float64,2},DataArray(airc))

ftcols=[1,2,3,5,6]
labcol = 4


features= airc[:,ftcols]

pcares = DimensionalityReduction.pca(features);
features=pcares.scores
labels = airc[:,labcol]

trInp = features[train,:]
trObs= labels[train]
tstInp = features[test,:]
tstObs = labels[test]

# modelsvm = svmtrain(trObs, trInp');
# (predicted_output, decision_values)=svmpredict(modelsvm, tstInp')

modelrf = DecisionTree.build_forest(trObs,trInp,4,50);
predrf=DecisionTree.apply_forest(modelrf,tstInp);

print("RMSE: ",rmse(tstObs,predrf))

data=convert(DataFrame,[predrf tstObs])
xs = 1:nrow(data)

Gadfly.set_default_plot_size(30cm, 10cm)
p1=plot(layer(data,x=xs,y=:x1,Geom.line),layer(data,x=xs,y=:x2,Geom.line,Theme(default_color=color("black"))))
p2=plot(layer(x=xs,y=data[:x1]-data[:x2],Geom.point),layer(x=0:nrow(data),y=rep(0,nrow(data)),Geom.line))
p3=plot(layer(x=data[:x1]-data[:x2],Geom.density))
hstack(p1,p2,p3)


