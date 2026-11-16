# Backscatter localization with low-complexity tags and multiple carrier waves: data and code

Raw measurement data and analysis code for the paper

> Y. Yamaguchi, V. Erdélyi, A. Uchiyama, "Feasibility of Backscatter Localization with Low-Complexity Tags Using
> Multiple Carrier Waves", submitted to *Sensors* (MDPI).

`reproduce.py` recomputes every figure and number in the paper's evaluation from the raw captures in `data/`.

## Quick start

    pip install -r requirements.txt
    python3 reproduce.py

A run takes under a minute on a multi-core machine. It writes `figures/*.png` and `results/` (see below). The
copies of both folders in this repository are the output of that command.

## Data

A transmitter sent 21 continuous-wave carriers spanning 30 MHz with equal spacing. The tag shifted them by 300 kHz.
A receiver with four antennas (half-wavelength spacing) captured the backscattered signal at 50 MHz. Both radios
were USRP X310s, synchronized by a shared 10 MHz and PPS reference.

    data/<environment>/21waves/<position>/samples.dat
    data/<environment>/21waves/<position>/samplingRates.txt

- `<environment>`: `abs` (RF absorber around the setup, multipath suppressed) or `noAbs` (natural multipath).
- `<position>`: tag position as distance from the receiver and angle, e.g. `150cm60deg`. Distances are 50, 100,
  150, 200, 250 and 300 cm; angles are 30, 60, 90, 120 and 150 degrees, giving 30 positions per environment.
- `wired`: a calibration capture with the transmitter cabled to all four receiver inputs through power splitters.
  It removes the phase offsets of the cables and receiver chains (see the paper, Section 3).

`samples.dat` holds the four receiver channels one after another. Each channel is a little-endian int64 sample
count `n` followed by `n` complex samples as interleaved little-endian float32 (real, imaginary). Each capture has
100,000 samples per antenna (2 ms). `samplingRates.txt` lists the transmitter and receiver sampling rates in Hz.
`load_capture()` in `reproduce.py` reads the format.

The 2-, 3-, 5- and 11-carrier results in the paper use subsets of the 21 received carriers; `CARRIER_SUBSETS` in
`reproduce.py` lists them.

## Results

- `results/estimates.csv`: every AoA and distance estimate (100 per position, for each carrier count) for the
  paper's 0-34.9 ns ToF search. This is Supplementary Table S1 of the paper.
- `results/numbers.txt` and `numbers.json`: median errors per carrier count, the effect of the ToF search limit
  (33, 35, 37 and 60 ns), and the share of estimates at the search limits. `numbers.json` has the exact values.
- `figures/`: the evaluation figures of the paper (Figures 11-14), with the paper's file names.

All analysis parameters are at the top of `reproduce.py` and match the paper.

## Software

The original analysis was written in MATLAB R2022b. The paper's results were computed with this Python port, using
Python 3.14, NumPy 2.4 and Matplotlib 3.10. Other Matplotlib versions give the same curves with slightly
different margins.

## License

The code is under the MIT License (`LICENSE`). The data in `data/`, `results/` and `figures/` are under the
Creative Commons Attribution 4.0 International License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
If you use them, please cite the paper.
