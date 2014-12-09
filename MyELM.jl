module MyELM
export ELM
export train
export predictVal
export predictClass
export sigmoid
export linear
export classify
export curvefit

type HiddenLayer
    weight_matrix::Matrix{Float64}
    bias_vector::Vector{Float64}
    function HiddenLayer(w::Matrix{Float64},bias::Vector{Float64})
        this = new()
        this.weight_matrix = w
        this.bias_vector = bias
        return this
    end
end

type OutputUnit
    output_weights::Matrix{Float64}
    class::Int64

    function OutputUnit(class::Int64)
        this = new()
        this.class = class
        this.output_weights = Array(Float64,0,0)
        return this
    end
end

type ELM
    output_layer::Vector{OutputUnit}
    hidden_layers::Array{HiddenLayer}
    hidden_output::Matrix{Float64}
    n_inputs::Int64
    v_hidden::Vector{Int64}
    n_outputs::Int64
    act_func::Function
    output_func::Function
    function ELM(n_inputs::Int64, v_hidden::Vector{Int64}, n_outputs::Int64, hidfn::Function,outputfn::Function=linear)
        this = new()
        this.n_inputs = n_inputs
        this.v_hidden = v_hidden
        this.n_outputs = n_outputs
        this.act_func = hidfn
        this.output_func = outputfn
        this.hidden_layers=Array(HiddenLayer,length(v_hidden))

        weight_matrix = rand(v_hidden[1], n_inputs) * 2 - 1
        bias_vector = rand(v_hidden[1])
        this.hidden_layers[1] = HiddenLayer(weight_matrix,bias_vector)
        for dim in 2:length(v_hidden)
            weight_matrix = rand(v_hidden[dim], v_hidden[dim-1]) * 2 - 1
            bias_vector = rand(v_hidden[dim])
            this.hidden_layers[dim] = HiddenLayer(weight_matrix, bias_vector)
        end

        this.output_layer = Array(OutputUnit,n_outputs)
        for i in 1:n_outputs
            this.output_layer[i] = OutputUnit(i)
        end
        return this
    end
end

function linear(x)
    return x
end

function sigmoid(x)
    # Sigmoid activation
    1 ./ (1 + exp(-x))
end

function classToBinaryMatrix(output::Vector{Int64},nclasses::Int64)
    # 2 class
    # outputs: 1 -> neuronOutput1:  1 0 1
    #          2    neuronOutput2:  0 1 0
    #          1
    data=zeros(Int64,nclasses,length(output))
    # check to make sure class starts from 0 

    output[indmin(output)] == 0? nothing: error("Assertion failed: Class assignment should start from 0")
    output[indmax(output)] == nclasses-1? nothing: error("Assertion failed: Class assignment max should be 1 less than number of classes")
    for i in 1:length(output)
        data[output[i]+1,i]=1
    end
    return data
end

function computeOutputWeights(model::ELM, outputs::Vector{Float64})
    # regression         

    for i in 1:length(model.output_layer)
        model.output_layer[i].output_weights = outputs' * pinv(model.hidden_output)
    end
end

function computeOutputWeights(model::ELM, outputs::Matrix{Int64})
    # outputs: each column j -> result of observation j
    #          each row i -> collection of results for output_layer i
    # Ex: 2 class
    # outputs: 1 -> neuronOutput1:  1 0 1
    #          2    neuronOutput2:  0 1 0
    #          1
    #if nprocs() < 2
    #    for i in 1:length(model.output_layer)
    #        model.output_layer[i].output_weights = outputs[i,:] * pinv(model.hidden_output)
    #    end
    #else
    res = @parallel (vcat) for i in 1:length(model.output_layer)
        (i,outputs[i,:] * pinv(model.hidden_output))
    end
    for i in 1:length(res)
        model.output_layer[i].output_weights = res[i][2]
    end
    #end
end

function train{T<:Real,S<:Real}(model::ELM,inputs::Matrix{T}, outputs::Vector{S})
    computeHidden(model,inputs)
    computeOutputWeights(model,outputs)
end

function train{T <: Real}(model::ELM,inputs::Matrix{T}, outputs::Vector{Int64})
    binaryMatrix = classToBinaryMatrix(outputs,model.n_outputs)
    computeHidden(model,inputs)
    computeOutputWeights(model,binaryMatrix)
end

function computeHidden{T <: Real}(model::ELM, inputs::Matrix{T})
    # inputs: each column j -> a feature
    #         each row i -> an observation

    (nObs,nFeatures) = size(inputs)

    hidden_output = zeros(length(model.hidden_layers[end].bias_vector), nObs)

    #for i = 1:nObs
    #     tmp = model.act_func(model.hidden_layer.weight_matrix * inputs[i, :]' + model.hidden_layer.bias_vector)
    #     hidden_output[:, i] = tmp
    #end

    hidden_output = Dict{Int64,Matrix{Float64}}()
    hidden_output[1] = model.act_func(model.hidden_layers[1].weight_matrix * inputs' .+ model.hidden_layers[1].bias_vector) 
    for i in 2:length(model.v_hidden)
        hidden_output[i] = model.act_func(model.hidden_layers[i].weight_matrix * hidden_output[i-1] .+ model.hidden_layers[i].bias_vector) 
    end
    model.hidden_output = hidden_output[hidden_output.count]
end

function predictClass(model::ELM, inputs::Matrix{Float64})
    computeHidden(model,inputs)
    nObs,nFeatures = size(inputs)
    pred = zeros(model.n_outputs  ,nObs)
    assert(size(model.hidden_output)[2]==nObs)
    for i in 1:length(model.output_layer)
        pred[i,:] = model.output_layer[i].output_weights * model.hidden_output
    end
    finalPred = zeros(Int64,size(pred)[2])
    assert(length(finalPred)==nObs)
    for i in 1:nObs
        finalPred[i]=indmax(pred[:,i])-1
    end
    return finalPred
end

function predictVal{T <: Real}(model::ELM, inputs::Matrix{T})
    computeHidden(model,inputs)
    nObs,nFeatures = size(inputs)
    pred = zeros(model.n_outputs,nObs)
    assert(size(model.hidden_output)[2]==nObs)
    for i in 1:length(model.output_layer)
        pred[i,:] = model.output_layer[i].output_weights * model.hidden_output
    end
    return vec(pred')
end

function rmse(x::Vector{Float64},y::Vector{Float64}) 
    return sqrt(mean((x-y).^2))
end

function classify(v_hidden::Array{Int64,1},trX::Matrix{Float64},trY::Vector{Int64},tstX::Matrix{Float64},tstY::Vector{Int64})
    nTrObs,nTrFts = size(trX)
    nOut = length(unique(trY))
    model = ELM(nTrFts,v_hidden,nOut,sigmoid)
    train(model,trX,trY)
    restr = predictClass(model,trX)
    restst = predictClass(model,tstX)
    println(mean(restr .== trY))
    println(mean(restst .== tstY))
    return model
end

function curvefit{T <: Real,S<:Real}(v_hidden::Array{Int64,1},trX::Matrix{T},trY::Vector{S},tstX::Matrix{T},tstY::Vector{S})
    nTrObs,nTrFts = size(trX)
    nOut = 1
    model = ELM(nTrFts,v_hidden,nOut,sigmoid)
    train(model,trX,trY)
    restr = predictVal(model,trX)
    restst = predictVal(model,tstX)
    println(rmse(restr,trY))
    println(cor(restr,trY))
    println(rmse(restst,tstY))
    println(cor(restst,tstY))
    return model
end
end
