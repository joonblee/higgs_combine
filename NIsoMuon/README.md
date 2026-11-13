
Full-spectrum review additions: `full_spectrum_fit.py --interpolate-signal`
now supports bracketed intermediate signal masses in local and Condor mode.
`full_spectrum_component_uncertainties.py M30/model.json` makes separate
QCD/DY/tt prefit relative-uncertainty plots from the compiled likelihood,
retaining era correlations. See `FULL_SPECTRUM_FIT.md` for executable commands,
interpolation assumptions and the distinction between prior responses and
postfit impacts. All 13 +/-5 sigma window boundaries are tabulated in
`sigfit_inputs/signal_windows_5sigma_native0p02.csv` (current coefficients,
0.02 GeV grid; model.json remains authoritative for other inputs).
