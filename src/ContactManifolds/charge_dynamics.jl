# Add at the beginning of charge_dynamics.jl if not already present
using DifferentialEquations
using LinearAlgebra


"""
    ChargeEvolutionProblem

Complete specification of charge evolution on contact manifold with transitions.
"""
struct ChargeEvolutionProblem{N}
    manifold::ContactManifold{N}
    reeb_flow::ReebFlow{N}
    initial_distribution::ChargeDistribution{N}
    transition_threshold::Float64
    simulation_time::Float64
end

"""
    evolve_charge_distribution!(problem::ChargeEvolutionProblem)

Main evolution function that handles both on-leaf evolution and transitions.
"""
function evolve_charge_distribution!(problem::ChargeEvolutionProblem{N}) where N
    current_dist = problem.initial_distribution
    current_time = 0.0
    
    trajectory = ChargeDistribution{N}[]
    transitions = NamedTuple{(:time, :from_leaf, :to_leaf, :force), 
                            Tuple{Float64, Vector{Float64}, Vector{Float64}, Float64}}[]
    
    while current_time < problem.simulation_time
        # Create current leaf
        current_leaf = SymplecticLeaf(current_dist.charges)
        
        # Evolve on current leaf
        remaining_time = problem.simulation_time - current_time
        leaf_evolution = evolve_on_leaf(
            current_leaf, 
            current_dist,
            problem.manifold.hamiltonian,
            (0.0, min(1.0, remaining_time))  # Evolve for at most 1 time unit
        )
        
        append!(trajectory, leaf_evolution)
        
        # Check for transition
        should_transition, force = detect_leaf_transition(
            leaf_evolution[end], 
            problem.transition_threshold
        )
        
        if should_transition && current_time + 1.0 < problem.simulation_time
            # Determine target leaf (simplified: random small charge transfer)
            new_charges = Vector(current_dist.charges)
            
            # Find maximal chemical potential difference
            max_i, max_j = 1, 2
            max_diff = 0.0
            for i in 1:N
                for j in i+1:N
                    diff = abs(current_dist.chemical_potentials[i] - 
                              current_dist.chemical_potentials[j])
                    if diff > max_diff
                        max_diff = diff
                        max_i, max_j = i, j
                    end
                end
            end
            
            # Transfer charge
            transfer_amount = 0.1 * min(abs(new_charges[max_i]), abs(new_charges[max_j]))
            if current_dist.chemical_potentials[max_i] > current_dist.chemical_potentials[max_j]
                new_charges[max_i] -= transfer_amount
                new_charges[max_j] += transfer_amount
            else
                new_charges[max_j] -= transfer_amount
                new_charges[max_i] += transfer_amount
            end
            
            # Create target leaf
            target_leaf = SymplecticLeaf(new_charges)
            
            # Record transition
            push!(transitions, (
                time = current_time + 1.0,
                from_leaf = Vector(current_dist.charges),
                to_leaf = new_charges,
                force = force
            ))
            
            # Perform transition
            transition_traj = transition_between_leaves(
                problem.reeb_flow,
                leaf_evolution[end],
                target_leaf,
                0.5  # Transition duration
            )
            
            append!(trajectory, transition_traj)
            current_dist = transition_traj[end]
            current_time += 1.5
        else
            current_dist = leaf_evolution[end]
            current_time += 1.0
        end
        
        if current_time >= problem.simulation_time
            break
        end
    end
    
    return trajectory, transitions
end

"""
    compute_electroneutrality_violation(trajectory)

Computes electroneutrality violation over time (should be ~0).
"""
function compute_electroneutrality_violation(trajectory::Vector{ChargeDistribution{N}}) where N
    violations = Float64[]
    
    for dist in trajectory
        violation = abs(sum(dist.charges))
        push!(violations, violation)
    end
    
    return violations
end