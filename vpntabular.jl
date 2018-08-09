@with_kw mutable struct VPNTabular
    ns::Int64 = 10
    na::Int64 = 4
    γ::Float64 = .9
    bbest::Int64 = 2
    d::Int64 = 4
    α::Float64 = .1 # learning rate
    Nsas::Array{Int64, 3} = zeros(Int64, ns, na, ns)
    Nsa::Array{Int64, 2} = zeros(Int64, na, ns)
    Q::Array{Float64, 2} = zeros(na, ns)
    V::Array{Float64, 1} = zeros(ns)
    R::Array{Float64, 2} = zeros(na, ns)
end
VPNTabular(kargs...) = VPNTabular(kargs...)
export VPNTabular

@inline function selectaction(learner::VPNTabular, policy, state)
    tempQ = learner.Q[:, state]
    temp_d = learner.d
    for i in 1:learner.na
        tempQ[i]= qplan(learner, state, i, temp_d)
    end
    return selectaction(policy, tempQ)
end

function qplan(learner::VPNTabular, s, a, d)
    s_next = sample(1:learner.ns, pweights(learner.Nsas[s, a, :]/learner.Nsa[a, s]))
    tempQ = collect(zip(learner.Q[:, s_next], collect(1:learner.na)))
    A = sort(tempQ, rev = true)[1:learner.bbest]
    if d == 0
        return learner.Q[a, s]
    elseif d == 1
        return learner.R[a, s] + learner.γ*learner.V[s_next]
    end
    q_0 = zeros(learner.bbest)
    for j in 1:size(A)[1]
        q_0[j] = qplan(learner, s_next, A[j][2], d-1)
    end
    return learner.R[a, s] + learner.γ * (1/d * learner.V[s_next] + (d-1)*maximum(q_0)/d)
end
function update!(learner::VPNTabular, buffer)
    s, a, r, s_next = buffer.states[end-1], buffer.actions[end-1],
        buffer.rewards[end], buffer.states[end]
    learner.R[a,s] += r
    learner.Nsa[a, s] += 1
    learner.Nsas[s, a, s_next] += 1
    learner.Q[a,s] += learner.α * (r + learner.γ*maximum(learner.Q[:, s_next]) - learner.Q[a,s])
    learner.V[s] = maximum(learner.Q[:, s])
end
