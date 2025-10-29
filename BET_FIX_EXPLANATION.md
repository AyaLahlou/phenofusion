# BET Phenology Detection Fix

## Problem Identified

The BET PFT was returning significantly fewer data points than BDT, NET, and NDT because of **overly restrictive date-based filtering logic**.

### Original Logic Issues

The original code had these problems:

1. **Backwards date checking**:
   ```python
   if doy_2 >= 50 and doy <= 150+buffer_days:
   ```
   This required:
   - End of window (`doy_2`) to be >= 50
   - Start of window (`doy`) to be <= 190

   This is overly restrictive! It would reject many valid phenological events.

2. **No slope verification**: The BET logic didn't check if there was actually a phenological signal (increasing/decreasing CSIF), unlike other PFTs.

3. **Incorrect window overlap**: The logic didn't properly check if the 30-day window overlaps with expected phenology periods.

## Solution Implemented

### 1. Added Slope-Based Signal Detection
Now BET uses the same signal detection as other PFTs:
```python
x = range(len(batch_df))
y = batch_df["CSIF"].values
has_signal = abs(y[0] - y[-1]) > min_diff

if not has_signal:
    continue

slope, _, _, _, _ = linregress(x, y)
is_sos = slope >= min_slope
is_eos = slope <= -min_slope - 0.0005
```

### 2. Fixed Window Overlap Logic
Changed from:
```python
if doy_2 >= 50 and doy <= 150+buffer_days:
```

To:
```python
if is_sos and doy >= 50 - buffer_days and doy_2 <= 150 + buffer_days:
```

This now:
- Verifies there's an actual SOS signal (`is_sos`)
- Checks if window START is after the expected range start (with buffer)
- Checks if window END is before the expected range end (with buffer)
- Allows the window to overlap properly with the expected phenology period

### 3. Added Debug Output
The code now reports:
- Total batches processed for BET
- Number of batches that matched criteria
- Match rate percentage

## Expected Results

With these changes, BET should now:
1. **Detect more valid phenological events** because the window overlap logic is correct
2. **Only detect events with actual signals** because of slope verification
3. **Be consistent with other PFTs** in methodology while respecting latitude-specific phenology timing

## How to Verify

Run the BET driver map generation and check the debug output:
```bash
sbatch bashscripts/drivers_BET.sh
```

Look for output like:
```
=== BET Debug Info ===
Total batches processed: XXXX
Batches matched (before filtering): YYYY
Match rate: ZZ.ZZ%
======================
```

Compare the number of SOS/EOS indices with other PFTs to ensure BET now produces comparable amounts of data.
