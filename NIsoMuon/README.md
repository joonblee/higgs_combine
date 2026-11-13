
Full-spectrum review additions: `full_spectrum_fit.py --interpolate-signal`
supports bracketed intermediate masses using the original counting-yield
interpolator in local and Condor mode. Fresh window-layout models normalise
r=1 to 25 events in the combined SR windows, retaining sideband leakage with
the same scale. No whole-histogram signal morph is used.
`full_spectrum_component_uncertainties.py M30/model.json` makes separate
QCD/DY/tt prefit relative-uncertainty plots from the compiled likelihood,
retaining era correlations. See `FULL_SPECTRUM_FIT.md` for executable commands,
interpolation assumptions and the distinction between prior responses and
postfit impacts. All 13 +/-5 sigma window boundaries are tabulated in
`sigfit_inputs/signal_windows_5sigma_native0p02.csv` (current coefficients,
0.02 GeV grid; model.json remains authoritative for other inputs).

For expected limits and r=0/r=1 impacts from completed full-spectrum workspaces,
run `full_spectrum_inference.py --results <r0-batch>/results --batch <new-batch>`.
It submits a complete Condor DAG, including plot jobs, independently of toys.
See [FULL_SPECTRUM_FIT.md](FULL_SPECTRUM_FIT.md#expected-limits-and-impacts-alongside-toys).
