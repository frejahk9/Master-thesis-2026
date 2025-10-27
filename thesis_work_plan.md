# Master Thesis Work Plan: AMOC Early Warning Signals
## Next 2-3 Weeks Implementation Guide

**Thesis Objective:** Encode climate models output into a low-dimensional latent space to identify early warning signals of AMOC tipping points.

**Current Status:** EOF analysis completed on SST, SSS, and MLD

**Available Data:** All CMIP6 variables including 3D ocean temperature, 3D salinity, mixed layer depth, and other variables

---

## Week 1: Validation & Model Stratification

### Day 1-2: Get AMOC Data & Stratify Models

**Goal:** Create your AMOC- and AMOC+ model groups

**Steps:**

1. Download AMOC strength data for CMIP6 historical runs (1900-2005)
   - Variable name: `msftmz` (meridional overturning streamfunction)
   - Look for maximum below 500m depth in North Atlantic (26°N is standard)

2. For each model:
   - Calculate annual mean AMOC strength
   - Compute linear trend (1900-2005)
   - Save: [model_name, AMOC_trend_Sv_per_century]

3. Create two lists:
   - `AMOC_minus_models` = [models with negative trend]
   - `AMOC_plus_models` = [models with positive trend]

4. Make a simple table/CSV with columns:
   - Model name | AMOC trend | Group (AMOC- or AMOC+)

**Code Structure:**
```python
# Pseudocode
for each model in CMIP6:
    AMOC_timeseries = load_variable('msftmz', model)
    AMOC_strength = max(AMOC_timeseries, where depth > 500m)
    trend = calculate_linear_trend(AMOC_strength, years=1900-2005)
    
    if trend < 0:
        AMOC_minus_models.append(model)
    else:
        AMOC_plus_models.append(model)
```

**Deliverable:** CSV file with all CMIP6 models classified by AMOC trend

---

### Day 3-4: Calculate FPI from Original (Full) Fields

**Goal:** Get the "ground truth" FPI before any compression

**Steps:**

1. Load historical SST data (1900-2005) for each model
   - Variable: `tos` (sea surface temperature)

2. Define the two regions (from Li & Liu, 2025):
   - Subpolar box: 46°N to 58°N, 49°W to 21°W
   - Gulf Stream box: 41°N to 45°N, 66°W to 40°W

3. For each model, for each year:
   - `SST_subpolar = spatial_mean(SST[subpolar_box])`
   - `SST_gulfstream = spatial_mean(SST[gulfstream_box])`
   - `FPI_SST_original[year] = SST_subpolar - SST_gulfstream`

4. Calculate linear trend of `FPI_SST_original` (1900-2005)

5. Repeat for SSS:
   - Variable: `sos` (sea surface salinity)
   - Same boxes, same calculation
   - Get `FPI_SSS_original`

**Code Structure:**
```python
# Define regions
subpolar_lat = [46, 58]  # degrees N
subpolar_lon = [-49, -21]  # degrees W
gulfstream_lat = [41, 45]
gulfstream_lon = [-66, -40]

# For each model
SST_subpolar = SST.sel(lat=slice(*subpolar_lat), 
                       lon=slice(*subpolar_lon)).mean(dim=['lat','lon'])
SST_gulfstream = SST.sel(lat=slice(*gulfstream_lat), 
                         lon=slice(*gulfstream_lon)).mean(dim=['lat','lon'])
FPI_SST = SST_subpolar - SST_gulfstream
```

**Deliverable:** 
- For each model: FPI_SST_original time series (1900-2005)
- For each model: FPI_SSS_original time series (1900-2005)
- Linear trends for both indices

---

### Day 5: Quick Sanity Check

**Goal:** Verify the article's main finding with your data

**Steps:**

1. Create a scatter plot:
   - X-axis: AMOC trend (for each model)
   - Y-axis: FPI_SST trend (for each model)

2. Calculate correlation:
   - `correlation = pearson_correlation(AMOC_trend, FPI_SST_trend)`

3. Expected result: 
   - Negative correlation (r ≈ -0.5 to -0.6)
   - Why negative? Weakening AMOC → more negative FPI_SST trend
   - This matches the article's finding (r = 0.57 with their sign convention)

4. Add reference lines:
   - Mark AMOC- models in one color
   - Mark AMOC+ models in another color

**Deliverable:** 
- One scatter plot showing AMOC vs FPI relationship
- Calculated correlation coefficient
- This confirms you can replicate the article's key finding!

---

## Week 2: Reconstruct & Validate

### Day 6-8: Reconstruct SST & SSS from Your EOFs

**Goal:** Use your existing EOF analysis to reconstruct the fields

**Steps:**

1. Review what you have from EOF analysis:
   - EOF_patterns_SST (spatial patterns/modes)
   - PC_SST (time series, principal components)
   - Same for SSS

2. Test reconstruction with different numbers of EOFs:
   - N = [5, 10, 15, 20, 30, 50]

3. For each N:
   ```python
   SST_reconstructed = sum(PC_SST[i] * EOF_pattern_SST[i] for i=1 to N)
   # Add back the mean if you removed it
   SST_reconstructed = SST_reconstructed + SST_mean
   ```

4. Compare with original:
   - Visual check: plot maps for a few sample years
   - Quantitative metrics:
     * Spatial correlation
     * Root Mean Square Error (RMSE)
     * Explained variance

5. Save reconstructed fields for the best N values

**Quality Metrics:**
- Spatial correlation > 0.95 = excellent
- Spatial correlation > 0.90 = good
- RMSE < 0.5°C for SST = good

**Deliverable:** 
- Table showing: N_EOFs | correlation | RMSE | explained_variance
- Decision on optimal number of EOFs

---

### Day 9-10: Calculate FPI from Reconstructed Fields

**Goal:** Can your compressed representation capture the AMOC fingerprint?

**Steps:**

1. Use your reconstructed SST and SSS fields

2. Apply same regional boxes as before:
   - Subpolar: 46-58°N, 49-21°W
   - Gulf Stream: 41-45°N, 66-40°W

3. Calculate FPI_SST_reconstructed and FPI_SSS_reconstructed

4. For each model, compare original vs reconstructed:
   - Plot time series of both FPI versions
   - Calculate correlation between them
   - Calculate trend of FPI_reconstructed
   - Compare trends: FPI_original_trend vs FPI_reconstructed_trend

5. Test different numbers of EOFs (N):
   - How many EOFs needed for correlation > 0.85?
   - How many for correlation > 0.90?
   - How many for correlation > 0.95?

**Key Question to Answer:**
Can I reconstruct the AMOC fingerprint from my compressed representation?

**Success Criteria:**
- Correlation between FPI_original and FPI_reconstructed > 0.85
- Trend difference < 15%

**Deliverable:** 
- Table: N_EOFs | FPI_correlation | trend_error
- Time series plots for 3-4 example models
- Decision: "I need N EOFs to adequately capture AMOC fingerprint"

---

### Day 11: Create Comparison Figure

**Goal:** Visual demonstration that your compression method works

**Steps:**

Create a comprehensive figure with 4 panels:

**Panel A:** AMOC trend vs FPI_SST trend (original data)
- Scatter plot of all models
- Show correlation coefficient
- Color by AMOC- (blue) and AMOC+ (red)

**Panel B:** AMOC trend vs FPI_SST trend (reconstructed with N EOFs)
- Same as Panel A but using reconstructed data
- Should look similar to Panel A

**Panel C:** FPI_original vs FPI_reconstructed (scatter)
- Each point is one model's FPI trend
- Perfect agreement = points on 1:1 line
- Show correlation coefficient

**Panel D:** Example time series (2 models)
- Time on x-axis (1900-2005)
- FPI_SST on y-axis
- Two lines per model: original (solid) and reconstructed (dashed)
- Choose one AMOC- and one AMOC+ model

**Deliverable:** 
- Single figure proving your compression preserves AMOC signal
- This becomes a key figure in your thesis!

---

## Week 3: Explore EOF Patterns & Multi-Model Composite

### Day 12-13: Examine Your EOF Spatial Patterns

**Goal:** Which EOF modes contain AMOC information?

**Steps:**

1. Plot spatial patterns of leading EOFs (modes 1-10) for SST:
   - Create maps showing EOF spatial structure
   - Look for dipole structures:
     * Cooling in subpolar region
     * Warming near Gulf Stream
   - Overlay the two FPI boxes on the maps
   - Note which modes show this pattern

2. For each EOF mode, calculate:
   - Correlation between PC[mode] and AMOC strength
   - Which modes correlate most strongly?
   - Is it the leading mode or a higher mode?

3. Repeat analysis for SSS EOFs:
   - Look for freshening/salinity dipole
   - Compare with SST patterns

4. Compare AMOC- vs AMOC+ model groups:
   - Calculate EOFs separately for each group
   - Do the leading modes differ?
   - Are AMOC fingerprints stronger in AMOC- models?

**Analysis Questions:**
- Is the AMOC fingerprint in EOF mode 1, or a higher mode?
- Do multiple modes contain AMOC information?
- Can I create a "AMOC mode" by combining several EOFs?

**Deliverable:** 
- Maps of leading EOF patterns with FPI boxes overlaid
- Table: EOF_mode | correlation_with_AMOC | explained_variance
- Identification of which modes are "AMOC-relevant"

---

### Day 14-15: Create AMOC- and AMOC+ Composites

**Goal:** Replicate article's Figure 3 with your data

**Steps:**

1. Create multi-model mean (MMM) for AMOC- models:
   ```python
   MMM_AMOC_minus_SST_trend = mean(SST_trend across all AMOC- models)
   MMM_AMOC_minus_SSS_trend = mean(SSS_trend across all AMOC- models)
   ```

2. Create MMM for AMOC+ models:
   ```python
   MMM_AMOC_plus_SST_trend = mean(SST_trend across all AMOC+ models)
   MMM_AMOC_plus_SSS_trend = mean(SSS_trend across all AMOC+ models)
   ```

3. Calculate difference (composite):
   ```python
   Difference_SST = MMM_AMOC_minus - MMM_AMOC_plus
   Difference_SSS = MMM_AMOC_minus - MMM_AMOC_plus
   ```

4. Plot these difference maps:
   - Should see cooling in subpolar Atlantic for SST difference
   - Should see warming near Gulf Stream for SST difference
   - Should see freshening in subpolar Atlantic for SSS difference
   - This is the "AMOC fingerprint"

5. Statistical significance:
   - Calculate t-test to show where differences are significant
   - Add stippling to map where p < 0.05

**Expected Result:**
Your maps should look similar to Figure 3 from Li & Liu (2025)

**Deliverable:** 
- Two maps: SST difference and SSS difference (AMOC- minus AMOC+)
- Figures showing the AMOC fingerprint clearly
- These validate that CMIP6 models show expected AMOC-driven patterns

---

### Day 16-17: Quick Exploration of MLD

**Goal:** See if mixed layer depth has AMOC signal

**Steps:**

1. Load MLD data:
   - Variable: `mlotst` or `mld` (mixed layer thickness)

2. Calculate MLD trends in the two FPI boxes:
   - Subpolar region trend
   - Gulf Stream region trend

3. Create MLD-based fingerprint:
   ```python
   FPI_MLD = MLD_trend(subpolar) - MLD_trend(gulfstream)
   ```

4. Check correlation with AMOC trend:
   - Scatter plot: AMOC_trend vs FPI_MLD
   - Calculate correlation coefficient
   - Is it significant?

5. If promising, test combined fingerprint:
   ```python
   FPI_combined = w1*FPI_SST + w2*FPI_SSS + w3*FPI_MLD
   # Start with equal weights: w1 = w2 = w3 = 1
   # Or optimize weights to maximize correlation with AMOC
   ```

6. Compare correlations:
   - AMOC vs FPI_SST alone
   - AMOC vs FPI_SSS alone
   - AMOC vs FPI_MLD alone
   - AMOC vs FPI_combined

**Bonus Analysis (if time):**
Test other combinations of your EOF modes as fingerprints

**Deliverable:** 
- Scatter plot of AMOC vs FPI_MLD
- Comparison table of different FPI variants
- Decision on whether to include MLD in final fingerprint

---

## End of Week 3: Summary for Supervisors

### What You Should Have Completed:

#### Data Products:
1. ✅ CSV file: All CMIP6 models with AMOC trends and classifications (AMOC- vs AMOC+)
2. ✅ Time series data: FPI (original and reconstructed) for all models
3. ✅ Optimal number of EOFs determined for reconstruction
4. ✅ EOF spatial pattern maps
5. ✅ AMOC- vs AMOC+ composite difference maps

#### Key Figures (5-6 figures ready):
1. **Figure 1:** Scatter plot - AMOC trend vs FPI_SST trend (original data)
2. **Figure 2:** Validation plot - FPI_original vs FPI_reconstructed
3. **Figure 3:** Time series examples (2-3 models showing original and reconstructed FPI)
4. **Figure 4:** EOF spatial patterns (leading 4-6 modes) with FPI boxes
5. **Figure 5:** Composite difference maps (AMOC- minus AMOC+)
6. **Figure 6:** Correlation table/heatmap - EOF modes vs AMOC strength

#### Key Findings to Report:
- "My EOF compression with **N modes** captures **X%** of FPI variance"
- "Correlation between AMOC and FPI in CMIP6 is **r = X** (consistent with Li & Liu 2025)"
- "EOF mode **Y** shows strongest AMOC fingerprint pattern"
- "I can reconstruct AMOC signal from compressed latent space with **X%** fidelity"
- "Multi-variable fingerprint [does/does not] improve AMOC detection"

---

## Practical Implementation Tips

### Start Small - Don't Process Everything at Once

```python
# Test your workflow with 3-5 models first:
test_models = ['CESM2', 'GFDL-ESM4', 'UKESM1-0-LL', 
               'MPI-ESM1-2-LR', 'CanESM5']

# Develop and debug code on these models
# Once code works smoothly, loop through all models
```

### Save Intermediate Results

Don't recalculate everything each time you run your code:

```python
import pickle
import numpy as np

# After calculating FPI for all models:
results = {
    'FPI_SST_original': FPI_SST_dict,
    'FPI_SSS_original': FPI_SSS_dict,
    'AMOC_trends': AMOC_trends_dict
}
pickle.dump(results, open('FPI_all_models.pkl', 'wb'))

# Load later:
results = pickle.load(open('FPI_all_models.pkl', 'rb'))
```

Or use NetCDF for larger datasets:
```python
import xarray as xr

# Save reconstructed fields
ds = xr.Dataset({
    'SST_reconstructed': SST_recon,
    'SSS_reconstructed': SSS_recon
})
ds.to_netcdf('reconstructed_fields_10EOFs.nc')
```

### Keep a Work Log

Create a simple text file or notebook tracking:
- What you did each day
- Which models have issues or missing data
- Parameter choices (number of EOFs, time periods, etc.)
- Problems encountered and solutions

### Organize Your Figures

Save figures with descriptive names:
```
fig1_AMOC_vs_FPI_original_data.png
fig2_FPI_validation_original_vs_reconstructed.png
fig3_timeseries_examples_CESM2_GFDL.png
fig4_EOF_patterns_SST_modes1-6.png
fig5_composite_AMOC_minus_vs_plus.png
```

---

## Troubleshooting Guide

### Problem: AMOC data is hard to find or access

**Solutions:**
- Start with published AMOC trends from literature:
  - IPCC AR6 report (Chapter 9)
  - Jackson et al. (2022) Nature Reviews Earth & Environment
  - Use Supplementary Tables from Li & Liu (2025)
- Alternative: Use density or sea level pressure differences as AMOC proxy
- Contact: CMIP6 data nodes often have derived variables available

### Problem: Too many models to process

**Solutions:**
- Start with 10-15 "tier 1" models known for good AMOC representation
- Focus on models used in Li & Liu (2025) - see their Supplementary Table
- Prioritize models with multiple ensemble members
- Expand to more models later once workflow is established

### Problem: Reconstruction quality is poor (correlation < 0.8)

**Solutions:**
- Try more EOFs (increase N)
- Check if you're comparing the right time periods
- Consider calculating EOFs for different regions separately
- Verify EOF calculation (check signs, normalization)
- May need nonlinear methods (autoencoder) - but try linear first!

### Problem: FPI doesn't correlate well with AMOC

**Solutions:**
- Double-check time period matches (1900-2005)
- Verify regional box definitions (longitude sign conventions: -180 to 180 or 0 to 360?)
- Confirm AMOC calculation is correct (maximum below 500m depth)
- Check for model outliers (some models may have biases)
- Try calculating FPI with detrended data (remove global warming signal)

### Problem: Missing data for some models

**Solutions:**
- Document which models have complete data
- Start with complete models only
- Note in your thesis which models were excluded and why
- Typical issues: different calendar types, missing years, different grid resolutions

### Problem: Code is too slow

**Solutions:**
- Use xarray with dask for lazy loading
- Process models in parallel (if you have multiple cores)
- Spatially subset data early (only load Atlantic region: 80°W to 20°E, 20°S to 70°N)
- Work with annual means first, monthly data later if needed
- Use coarser temporal resolution for initial tests

---

## Quick Reference: CMIP6 Variables

### Variables You Need:

**For Week 1-3:**
- `tos`: Sea surface temperature [K or °C]
- `sos`: Sea surface salinity [psu or 0.001]
- `mlotst` or `mld`: Mixed layer thickness [m]
- `msftmz`: Ocean meridional overturning mass streamfunction [kg/s]

**For Later (Week 4+):**
- `thetao`: Sea water potential temperature (3D) [K or °C]
- `so`: Sea water salinity (3D) [psu or 0.001]
- `zos`: Sea surface height [m]
- `hfds`: Downward heat flux at sea surface [W/m²]

### CMIP6 Experiment Names:
- `historical`: Historical simulations (typically 1850-2014)
- `piControl`: Pre-industrial control runs
- `ssp585`: Future scenario (for later analysis)

### Where to Access Data:
- ESGF nodes (Earth System Grid Federation)
- DKRZ (German Climate Computing Center)
- CEDA/JASMIN (UK)
- Local institutional data archives

---

## Success Criteria - End of Week 3

By the end of these three weeks, you should be able to confidently state:

### Primary Achievement:
> "I can compress CMIP6 climate data using EOF analysis and still capture the AMOC fingerprint with >85% fidelity. My latent space (compressed representation) preserves AMOC-relevant information, validating that I can use it for early warning signal detection in subsequent work."

### Specific Metrics:
- ✅ Classified 40+ CMIP6 models by AMOC behavior
- ✅ Calculated AMOC fingerprint indices for all models
- ✅ Determined optimal EOF truncation (N modes)
- ✅ Achieved FPI reconstruction correlation > 0.85
- ✅ Identified which EOF modes contain AMOC signal
- ✅ Reproduced key findings from Li & Liu (2025)

### Next Steps Preview:
With this foundation established, Week 4+ will focus on:
- Applying early warning signal metrics (AR1, variance, DFA) to principal components
- Detecting critical slowing down in AMOC- models
- Testing whether latent space changes predict AMOC slowdown
- Potentially implementing nonlinear compression (autoencoder)

---

## Summary Checklist

Use this to track your progress:

**Week 1:**
- [ ] Downloaded AMOC data for CMIP6 models
- [ ] Calculated AMOC trends (1900-2005)
- [ ] Created AMOC- and AMOC+ model lists
- [ ] Calculated FPI_SST and FPI_SSS from original fields
- [ ] Created scatter plot: AMOC vs FPI
- [ ] Confirmed correlation matches literature (~0.5-0.6)

**Week 2:**
- [ ] Reconstructed SST and SSS from EOFs
- [ ] Tested multiple EOF truncations (N = 5, 10, 15, 20, 30, 50)
- [ ] Calculated FPI from reconstructed fields
- [ ] Compared original vs reconstructed FPI
- [ ] Determined optimal number of EOFs
- [ ] Created 4-panel validation figure

**Week 3:**
- [ ] Plotted spatial patterns of leading EOFs
- [ ] Calculated EOF-AMOC correlations
- [ ] Identified AMOC-relevant EOF modes
- [ ] Created AMOC- and AMOC+ composite differences
- [ ] Tested MLD-based fingerprint
- [ ] Explored multi-variable fingerprint combinations
- [ ] Prepared summary for supervisors

**Documentation:**
- [ ] Maintained work log with daily notes
- [ ] Saved all figures with clear names
- [ ] Created data dictionaries for outputs
- [ ] Documented model issues/exclusions
- [ ] Prepared presentation slides (optional but recommended)

---

## Additional Resources

### Key Papers to Reference:
1. **Li & Liu (2025)** - Communications Earth & Environment
   - Your primary methodological reference
   - Use their fingerprint index definitions

2. **Caesar et al. (2018)** - Nature
   - Original AMOC fingerprint paper
   - Alternative index definition for comparison

3. **Jackson et al. (2022)** - Nature Reviews Earth & Environment
   - Recent review of AMOC evolution
   - Good for AMOC trends in CMIP6

4. **Boers (2021)** - Nature Climate Change
   - Early warning signals for AMOC
   - Inspiration for your Week 4+ work

### Python Packages You'll Use:
```python
import xarray as xr  # for NetCDF data handling
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plotting
import cartopy  # for maps
from scipy import stats  # for correlations, t-tests
import pandas as pd  # for organizing results
```

### Useful Code Snippets Repository:
Consider creating a `utils.py` file with common functions:
```python
def calculate_FPI(SST_field, subpolar_box, gulfstream_box):
    """Calculate AMOC fingerprint index"""
    # Your implementation
    
def reconstruct_from_EOFs(PCs, EOF_patterns, n_modes):
    """Reconstruct field from EOF decomposition"""
    # Your implementation
    
def plot_fingerprint_map(data, title):
    """Standard map for AMOC fingerprint"""
    # Your implementation
```

---

## Questions for Your Supervisors

Prepare these questions for your next meeting:

### Data & Methods:
1. Do we have access to 3D subsurface temperature/salinity for all CMIP6 models I'm analyzing?
2. Should I use all available CMIP6 models (~50+) or focus on a subset with better AMOC representation?
3. What time period should I prioritize: 1900-2005 (like Li & Liu) or extend to 2014/2023?

### Analysis Choices:
4. What threshold for FPI reconstruction quality should I aim for? (currently targeting >0.85 correlation)
5. Should I complete full validation before moving to early warning signals, or start parallel work?
6. Do you want me to compare my latent space approach against calculating FPI directly (without compression)?

### Scope & Timeline:
7. After validation (Week 3), should I focus on:
   - Critical slowing down metrics on EOF principal components? OR
   - Implementing autoencoder for nonlinear compression? OR
   - Extending to 3D subsurface analysis?

8. How detailed should my model-by-model analysis be vs. focusing on multi-model means?

### Thesis Structure:
9. Should my thesis have separate validation chapter before early warning chapter?
10. Do you want preliminary results presented at [upcoming conference/group meeting]?

---

**Document Version:** 1.0  
**Last Updated:** October 27, 2025  
**Author:** [Your Name]  
**Supervisors:** [Supervisor Names]

---

*Good luck with your analysis! Remember: start small, validate thoroughly, and build incrementally. The goal of these three weeks is validation, not perfection. You're building the foundation for your early warning signal detection work.*