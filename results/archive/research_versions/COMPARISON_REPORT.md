# Baseline PINN vs Multi-Scale PINN (No Fourier)

Fair A/B comparison with seed 42, 5000 epochs, Adam, learning rate 1e-3, and identical training/evaluation points per problem. Only the architecture changes.

| Problem | Model | Params | Final Loss | Relative L2 | MSE | Time (s) |
|---|---|---:|---:|---:|---:|---:|
| decay | Baseline PINN | 2209 | 1.471854e-07 | 4.862147e-05 | 1.022374e-09 | 4.68 |
| decay | Multi-Scale PINN (No Fourier) | 2940 | 7.226959e-07 | 4.366501e-04 | 8.245574e-08 | 12.55 |
| oscillator | Baseline PINN | 2209 | 4.043397e-07 | 3.973069e-05 | 9.383073e-10 | 8.44 |
| oscillator | Multi-Scale PINN (No Fourier) | 2940 | 2.169664e-06 | 6.128259e-05 | 2.232375e-09 | 22.89 |
| burgers | Baseline PINN | 2241 | 1.587503e-02 | 1.900072e-01 | 1.339885e-02 | 26.54 |
| burgers | Multi-Scale PINN (No Fourier) | 2988 | 9.080513e-03 | 2.455006e-01 | 2.236831e-02 | 57.51 |

## Result

The current no-Fourier Multi-Scale architecture did **not** outperform the baseline PINN for these three single-seed experiments. This is an experimental result, not hidden or tuned away.
