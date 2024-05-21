file functionality:
1. data_generation.py -- use this file to generate time-dependent states for quantum/classical systems.
2. helper_function.py -- provide function to generate topologies.
3. persistence_analysis.py -- file to calculate persistence probability and save the calculations (only support Crank-Nicolson algorithm, function read_phi should be modified to support read evolution data for eigenvalue decomposition approach.) 
4. plot_dpp.py -- file to plot (diffusive) persistence probability distribution.
5. plot_variance_state.py -- file to plot variance (two-point correlation, autocorrelation)of density and phase.
6. plot_rho_t.py -- file to plot time-dependent evolution, and heatmaps of density/phase.
7. plot_IPR.py -- file to study localization and plot Inverse Participation Ratio. 
8. plot_state_distribution.py -- file to plot distribution of density/phase.
9. plot_entropy.py -- file to plot entropy of the system.
10. plot_sign_time.py -- file to plot distribution of sign (+/-) switch.
