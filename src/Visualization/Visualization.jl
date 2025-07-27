module Visualization

using GLMakie
using Colors
using ColorSchemes
using GeometryBasics
using LinearAlgebra
using Statistics

# Import from parent module
using ..ContactManifolds
using ..Utils

export visualize_contact_manifold, animate_charge_evolution
export plot_symplectic_leaves, plot_reeb_flow, plot_phase_portrait
export create_morphogenetic_field_visualization
export plot_tropical_computation, plot_hierarchical_structure

include("manifold_plots.jl")

end # module Visualization