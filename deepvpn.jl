# Here we implement deep value prediction networks (VPN) as in (1) with asynchronous
# n-step Q-learning with the ReinforcementLearning.jl (2) and Flux.jl (3).
# (See 'citations.txt' for full citations, links, and details.)
#
# Run this code as a ReinforcementLearning.jl 'learner' instance.


# ------------------- INITIALIZE OBJECTS, SET PARAMETERS ----------------------#
struct ActionDense{Tw, Tb}
    # ---Create abstract layer structure---
    w::Tw
    b::Tb
    f::Function
end

function ActionDense(na, nin, nout, f)
    # ---Create function for conversion of abstract states to actions---
    ActionDense([Flux.param(rand(nout, nin)) for _ in 1:na], [Flux.param(rand(nout)) for _ in 1:na], f)
end

export ActionDense
(l::ActionDense)(a, x) = l.f(l.w[a] * x + l.b[a]) # Initialize linear conversion layer
(c::Chain)(a, x) = foldl((x, m) -> m(x), c.layers[1](a, x), c.layers[2:end]) # Create chain of layers

@with_kw mutable struct deepVPN{Tfenc, Tfval, Tftrans, Tfout, ToptT, Topt}
    # ----Environment/Net Params----
    na::Int64 # dimensionality of actions
    ns::Int64 # dimensionality of states
    nabs::Int64 # number of abstract states - corresponds to the output dimension of fenc

    # ----Net---- (composed of four modules)
    fenc::Tfenc # encodes state into an 'abstract state'
    fval::Tfval # finds 'value' of state from 'abstract state'
    ftrans::Tftrans # finds next state from 'abstract state'-action pair
    fout::Tfout # predicts reward from 'abstract state'-action pair

    # ----Target Net---- (same structure as Net, following DDQN convention (4))
    tfenc::Tfenc = deepcopy(fenc)
    tfval::Tfval = deepcopy(fval)
    tftrans::Tftrans = deepcopy(ftrans)
    tfout::Tfout = deepcopy(fout)

    # ----General Parameters----
    t::Int64 = 0 # time tracker
    γ::Float64 = .9 # discount rate
    η::Float64 = 2.5e-4 # learning rate
    ρ::Float64 = 0.95 # gradient momentum for RMSProp
    n::Int64 = 5 # 'n' for n-step Q learning
    M::Int64 = 10^3 # number of (s,a) pairs from random policy stored
    loss::Flux.Tracker.TrackedReal{Float64} = param(0.)
    targetupdateperiod::Int64 = 500 # periodicity of target net updates w.r.t time tracker, 't'
    params::Array{Any, 1} = vcat(map(Flux.params, [fenc, ftrans, fval, fout])...) # concatenate all net params into single array for backprop
    opttype::ToptT = Flux.RMSProp # set optimizer type
    opt::Topt = opttype(params(fenc) ∪ params(fval) ∪ params(ftrans) ∪ params(fout), η; ρ = ρ) # initialize optimizer

    # ----VPN Q-Learning/Planning Parameters----
    # - Note: to calculate accumulated n-step q learning losses, VPN uses n
    # ('s', 'actions') tuples where 'actions' is an array of the k actions
    # taken after being in state 's'. This is why learning starts at t =
    # 'M + k' rather than simply at t = 'M' (1).
    bbest::Int64 = na
    Q1 = []
    Q2 = []
    d::Int64 = 4 # (- Note: must be greater than or equal to 1)
    k::Int64 = d-1
    memory::Array{Any} = []
    expandedbuffer::Array{Any} = zeros(k+n+1)
    ts_before_online_learning = 5*10^5 # Number of timesteps (ts) at which offline learning stops, online learning starts

    # ----Exploration Parameters----
    ϵ_START::Float64 = 0.5  # Initial exploration rate
    ϵ_STOP::Float64 = 0.1    # Final exploratin rate
    ϵ_STEPS::Int64 = 1000000   # Final exploration frame, using linear annealing
    learningstarted::Bool = false

end
export deepVPN

# Create deepVPN instance, initializing all net and target net modules
deepVPN(fenc, fval, ftrans, fout; kargs...) = deepVPN(; fenc = Flux.gpu(fenc),
fval = Flux.gpu(fval), ftrans = Flux.gpu(ftrans), fout = Flux.gpu(fout), kargs...)

using Flux, StatsBase
import Flux.params, StatsBase.entropy


# ---------------------------- ACTION SELECTION ----------------------------#

@inline function selectaction(learner::deepVPN, policy, state)
    # Choose an action randomly before learning starts or if
    # rand() < get_ϵ(learner) during the 'exploration period'. Otherwise choose
    # action with the maximal Q-value calculated with depth=d (1).
    if ~learner.learningstarted && (learner.t - learner.ts_before_online_learning) > 0 # Check if learning period has ended
        println("-------------------NEURAL NET--------------------")
        learner.learningstarted = true
    end
    if (f < learner.ϵ_STEPS && rand() < get_ϵ(learner)) || (f < 0)
        return sample(1:learner.na)
    else
        Q = zeros(learner.na)
        for i in 1:learner.na
            Q[i] = getq(learner, state, i, learner.d, false)
            if i ==1
                push!(learner.Q1, Q[i])
            else
                push!(learner.Q2, Q[i])
            end
        end
        return findmax(Q)[2]
    end
end

function get_ϵ(learner::deepVPN)
    # Find ϵ for current frame via linear annealing, keeping ϵ constant after
    # calculated value drops below 'learner.ϵ_STOP'.
    f = learner.t - learner.ts_before_online_learning
    start, steps, stop = learner.ϵ_START, learner.ϵ_STEPS, learner.ϵ_STOP
    current_linear_ϵ = start + f * (stop - start)/steps
    if current_linear_ϵ > learner.ϵ_STOP
        return current_linear_ϵ
    else
        return learner.ϵ_STOP
    end
end


# ----------------- VPN Q VALUE CALCULATION/K-STEP PREDICTION------------------#

function getq(learner::deepVPN, s, a, d, calculatinglosses::Bool)
    # INPUT: learner, s [= (float vector) a single state], a [= (int) a single action],
    # d [= depth], calculatinglosses [= whether or not method is being called during planning]
    #
    # OUTPUT: float corresponding to the calculated Q-value of the state-action pair
    # with depth d, following convention introduced in (1).
    #
    # - Note: As in (1), for planning, VPN uses the 'net'. When accumulating q-losses
    # however, Q-values are calculated based off of the 'target net' instead.
    if ~calculatinglosses
        fenc, fout, ftrans, fval = learner.fenc, learner.fout, learner.ftrans, learner.fval
    else
        fenc, fout, ftrans, fval = learner.tfenc, learner.tfout, learner.tftrans, learner.tfval
    end
    abstract_s = fenc(s)
    s_next = ftrans(a, abstract_s)
    r = fout(a, abstract_s).data[1]
    v = fval(s_next).data[1]
    if d == 1
        return r + learner.γ * v
    else
        # Calculate Q-values for each action in abstract_next_state
        tempq = []
        for i in 1:learner.na
            temp_s_next = ftrans(i, s_next)
            vtemp = fval(temp_s_next).data[1]
            rtemp = fout(i, s_next).data[1]
            push!(tempq, (rtemp + learner.γ * vtemp, i)) # ([= (float) Q-value], [= (int) corresponding action])
        end
        # Find actions corresponding to bbest maximal q values, d=1
        A = sort(tempq, rev=true)[1:learner.bbest]
        q_0 = zeros(learner.bbest)
        for i in 1:length(q_0)
            q_0[i] = getq(learner, s, A[i][2], d-1, calculatinglosses) # Get Q-values
        end
        return r + learner.γ * ((1/d) * v + (d-1)/d * maximum(q_0)) # Return final d depth Q-value, updated with discount factor
    end
end


function kstepprediction(learner::deepVPN, s, a, onlyr::Bool)
    # INPUT: learner, s [= (float vector) single start state], a [= (int vector) array of length k actions],
    # onlyr [= (bool) representing whether to only return kstep r prediction or kstep v and r predictions]
    #
    # OUTPUT: if onlyr -> one array of length k corresponding to k-step predicted rewards.
    # if ~onlyr -> 2-tuple with two arrays of length k corresponding to k-step predicted values and predicted rewards
    r_l, v_l = [], []
    currentabsstate = learner.fenc(s)
    if onlyr
        for i in 1:length(a)
            push!(r_l, learner.fout(a[i], currentabsstate))
            currentabsstate = learner.ftrans(a[i], currentabsstate)
        end
        return r_l
    else
        for i in 1:length(a)
            push!(r_l, learner.fout(a[i], currentabsstate))
            push!(v_l, learner.fval(currentabsstate))
            currentabsstate = learner.ftrans(a[i], currentabsstate)
        end
        return v_l, r_l
    end
end


# ------------------------- UPDATE LOSSES/MEMORY ------------------------------#

function remember!(learner::deepVPN, buffer)
    # Updates VPN's memory buffer with the 'k' latest (s, a, r, s') tuples from temp-memory.
    l = length(learner.expandedbuffer)
    gatheractions = Int64[]
    for i in l - learner.k:l-1
        push!(gatheractions, learner.expandedbuffer[i][2])
    end
    push!(learner.memory, (learner.expandedbuffer[l-learner.k][1],
            gatheractions, learner.expandedbuffer[l-learner.k][3], learner.expandedbuffer[l-learner.k+1][1]))
end

square(x) = dot(x, x) # Overload Julia's 'square' function for vectors

@inline function accumulateqlosses!(learner::deepVPN)
    # Calculate k-step Q-value losses for backprop following convention in (1).
    # Update the cumulative loss [= learner.loss].
    l = length(learner.expandedbuffer)
    states, actions, rewards, states_mask = [], [], [], []
    for i in l-learner.n-learner.k+1:l-learner.k
        push!(states, learner.expandedbuffer[i][1])
        push!(actions, learner.expandedbuffer[i][2])
        push!(rewards, learner.expandedbuffer[i][3])
        push!(states_mask, learner.expandedbuffer[i][4])
    end
    for i in l-learner.k+1:l
        push!(actions, learner.expandedbuffer[i][2])
    end
    loss = 0
    R = 0
    for j in length(states):-1:1
        s, r, sm = states[j], rewards[j], states_mask[j]
        a = actions[j:j+learner.k-1]
        if j == length(states)
            Q = zeros(learner.na)
            for i in 1:learner.na
                Q[i] = getq(learner, s, i, learner.d, true)
            end
            R += r + learner.γ .* maximum(Q) .* sm
        else
            R = r + learner.γ * R
        end
        v_l, r_l = kstepprediction(learner, s, a, false)
        loss += sum(square.(R .- v_l) + square.(r .- r_l))
    end
    learner.loss += loss
end

@inline function replaymemory!(learner::deepVPN)
    # Sample 'n' (s, a, r, s') tuples from VPN's memory buffer, adding the corresponding
    # losses to the cumulative loss [= learner.loss] for backprop.
    i = StatsBase.sample(1:length(learner.memory)-learner.n, 1, replace = false)[1]
    loss = 0
    for j in 0:learner.n-1
        if j == 0
            s, a, r, s_next = learner.memory[i+j]
        else
            s, a, r, s_next = learner.memory[i+j]
        end
        r_l = kstepprediction(learner, s, a, true)
        loss += sum(square.(r .- r_l))
    end
    learner.loss += loss
end

function update!(learner::deepVPN, b)
    # Global VPN update function, following conventions of learner instances in (2).
    #
    # - Note: Offline learning begins only when expandedbuffer has been filled (1).
    learner.t += 1 # Increment time
    if learner.t % learner.targetupdateperiod == 0 # Check whether update of target net is due
        learner.tfenc = deepcopy(learner.fenc)
        learner.tftrans = deepcopy(learner.ftrans)
        learner.tfval = deepcopy(learner.fval)
        learner.tfout = deepcopy(learner.fout)
    end
    if learner.t <= length(learner.expandedbuffer) # Before offline learning begins, fill expandedbuffer
        learner.expandedbuffer[learner.t] = (b.states[end-1], b.actions[end-1],
            b.rewards[end], b.done[end])
    else # Run learning scheme: update the expandedbuffer, accumulate losses, run backprop, update all parameters
        l = length(learner.expandedbuffer)
        for i in 2:1:l
            learner.expandedbuffer[i-1] = learner.expandedbuffer[i]
        end
        learner.expandedbuffer[l] = (b.states[end-1], b.actions[end-1],
            b.rewards[end], b.done[end])
        if length(learner.memory) < learner.M # Check if random action memory has been filled
            remember!(learner, b)
        elseif learner.t % learner.n == 0
            accumulateqlosses!(learner) # Increment learner.losses with q losses
            replaymemory!(learner) # Increment learner.losses with losses from RP memory
            Flux.back!(learner.loss) # Backpropagate/calculate gradients
            learner.opt() # Run RMSProp
            learner.loss = 0 # Reset learner.losses
        end
    end
end
