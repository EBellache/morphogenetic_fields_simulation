using MorphogeneticFields
using GLMakie

# Set up a 3-compartment system
N = 3

# Define Hamiltonian (simplified energy function)
function hamiltonian(charges, phases, t)
    # Kinetic energy from phases
    kinetic = sum(phases.^2) / 2
    
    # Potential energy from charge imbalance
    potential = sum(charges.^2) / 2
    
    # Coupling energy
    coupling = 0.0
    for i in 1:length(charges)-1
        coupling += charges[i] * charges[i+1] * cos(phases[i] - phases[i+1])
    end
    
    return kinetic + potential - 0.5 * coupling
end

# Create contact manifold
manifold = ContactManifold{N}(hamiltonian)

# Set up conductivity matrix for Reeb flow
σ = [0.0  0.1  0.05;
     0.1  0.0  0.1;
     0.05 0.1  0.0]

reeb_flow = ReebFlow(manifold, σ, 300.0)

# Initial charge distribution (must satisfy electroneutrality)
initial_charges = [0.5, -0.3, -0.2]
initial_phases = [0.0, π/4, π/2]
initial_μ = [1.0, 0.5, 0.7]

initial_dist = ChargeDistribution(initial_charges, initial_phases, initial_μ)

# Create evolution problem
problem = ChargeEvolutionProblem(
    manifold,
    reeb_flow,
    initial_dist,
    0.3,  # transition threshold
    10.0  # simulation time
)

# Simulate evolution
println("Simulating charge distribution evolution...")
trajectory, transitions = evolve_charge_distribution!(problem)

println("Number of transitions: $(length(transitions))")
for (i, trans) in enumerate(transitions)
    println("Transition $i at t=$(trans.time): force=$(trans.force)")
end

# Visualize results
println("Creating visualization...")
fig = visualize_contact_manifold(trajectory, transitions)
display(fig)

# Create animation
println("Creating animation...")
animate_charge_evolution(trajectory, transitions, filename="charge_evolution.mp4")

println("Done! Check charge_evolution.mp4 for the animation.")