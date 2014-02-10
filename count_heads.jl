function count_heads(n)
    c::Int = 0
    for i=1:n
        c += randbool()
    end
    c
end
