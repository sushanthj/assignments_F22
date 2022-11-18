# Good questions to ask

1. Why do we need rodrigues form of rotation here?
2. How to get M2? what's going on behind this optimizer?
3. Why do we flatten the x?

# Implementation Observations

## Optimization methods used in scipy.optimize.minimize
1. Using the default method, reprojection error came down to 10.52
2. Using Nelder-Mead the error was at 72
3. Using Powell the error was at 10.54 (computed very quickly as well ~ 2.9s)
4. Using L-BFGS-B, the error was 57.64
5. Using TNC, the error was 10.55 but it took too long (33.8s)