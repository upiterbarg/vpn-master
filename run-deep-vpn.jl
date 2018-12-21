__precompile__()

module deepVPN

using DataStructures, Parameters
include("deepvpn.jl")
for (root, dirs, files) in walkdir(joinpath(@__DIR__, "learner"))
    for file in files
        if splitext(file)[end] == ".jl"
#             println("including $(joinpath(root, file)).")
            include(joinpath(root, file))
        end
    end
end


end # module
