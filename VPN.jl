__precompile__()

module VPN

using DataStructures, Parameters
include("vpntabular.jl")
for (root, dirs, files) in walkdir(joinpath(@__DIR__, "learner"))
    for file in files
        if splitext(file)[end] == ".jl"
#             println("including $(joinpath(root, file)).")
            include(joinpath(root, file))
        end
    end
end


end # module
