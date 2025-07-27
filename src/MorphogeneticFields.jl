module MorphogeneticFields

using LinearAlgebra
using StaticArrays
using DifferentialEquations
using ForwardDiff
using Manifolds

# First, include and load the Utils module
include("Utils/Utils.jl")
using .Utils

# Then include ContactManifolds which depends on Utils
include("ContactManifolds/ContactManifolds.jl")
using .ContactManifolds

# Finally include Visualization which depends on both
include("Visualization/Visualization.jl")
using .Visualization

# Export main types and functions
export ContactManifold, SymplecticLeaf, ReebFlow
export ChargeDistribution, ElectroneutralityConstraint, ChargeEvolutionProblem
export evolve_charge_distribution!, compute_reeb_vector_field
export visualize_contact_manifold, animate_charge_evolution

end # module