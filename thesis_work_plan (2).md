# Master Thesis Work Plan: AMOC Early Warning Signals via EOF Analysis
## Next 3 Weeks Implementation Guide

**Thesis Objective:** Encode climate models output into a low-dimensional latent space to identify early warning signals of AMOC tipping points. By reducing the complex climate data to a few hundred key latent variables, the project seeks to detect subtle temporal shifts in this latent space that may indicate an impending AMOC tipping event.

**Approach:** Compress the climate state into a smaller dimension space using EOF analysis. From this smaller space, reconstruct AMOC values similar to those obtained with full climate state (before compression). Then identify patterns in the smaller latent space to explain when/how the AMOC slows down.

**Current Status:** 
- ✅ EOF analysis completed on SST, SSS, and MLD for one ensemble member in one model
- Variables: Temperature and Salinity (SST, SSS initially; 3D later)
- Time period: 1840-2014

**Available Data:** 
- CMIP6 model data (multiple ensemble members available)
- Observational data: EN4, etc.
- All standard ocean variables including 3D temperature and salinity

---

## Week 1: Multi-Member EOF Analysis & Ensemble Mean

### Day 1-2: Expand EOF Analysis to Multiple Ensemble Members

**Goal:** Perform EOF analysis on multiple ensemble members of the same model

**Steps:**

1. **Select your model and members:**
   ```python
   # Example: if using CESM2
   model = 'CESM2'
   members = ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1']
   # Aim for at least 5 members, ideally 10 if available
   ```

2. **Load data for each member (1840-2014):**
   - SST (`tos`)
   - SSS (`sos`)
   - MLD (`mlotst` or `mld`)

3. **Perform EOF analysis on each member separately:**
   ```python
   for member in members:
       # Load data
       SST = load_data(model, member, 'tos', years='1840-2014')
       
       # Remove climatology (seasonal cycle)
       SST_anom = SST - SST.groupby('time.month').mean('time')
       
       # Perform EOF
       eof_patterns[member], pcs[member], explained_var[member] = calculate_eof(SST_anom, n_modes=10)
       
       # Save results
       save_eof_results(member, eof_patterns[member], pcs[member])
   ```

4. **Repeat for SSS and MLD**

5. **Document:**
   - Number of members analyzed
   - Time period for each member
   - Any missing data or issues

**Quality Checks:**
- Do all members have complete data (1840-2014)?
- Are EOF patterns physically reasonable?
- Check explained variance: does EOF1 explain >20% for SST?

**Deliverable:** 
- EOF patterns and PCs for all members
- Table: Member | EOF1_variance | EOF2_variance | EOF3_variance (for each variable)

---

### Day 3-4: Compare Leading Modes Across Members

**Goal:** Compare the first 3 EOF modes across all ensemble members

**Steps:**

1. **Visualize EOF spatial patterns for each member:**
   ```python
   # For SST, plot EOF modes 1-3 for all members
   fig, axes = plt.subplots(n_members, 3, figsize=(15, 4*n_members))
   
   for i, member in enumerate(members):
       for mode in [0, 1, 2]:  # EOF 1, 2, 3
           plot_eof_pattern(eof_patterns[member][mode], ax=axes[i, mode])
           axes[i, mode].set_title(f'{member} - EOF{mode+1}')
   ```

2. **Compare spatial patterns:**
   - Are EOF1 patterns similar across members?
   - Do they capture the same physical features?
   - Calculate pattern correlation between members:
     ```python
     pattern_corr = correlate_2d_fields(EOF1_member1, EOF1_member2)
     ```

3. **Compare principal components (time series):**
   ```python
   # Plot PC1 for all members on same plot
   plt.figure(figsize=(12, 6))
   for member in members:
       plt.plot(time, pcs[member][0], label=member, alpha=0.7)
   plt.legend()
   plt.title('PC1 comparison across members')
   ```

4. **Calculate correlation between PC time series:**
   ```python
   # Correlation matrix of PC1 across all members
   pc1_correlations = calculate_pc_correlation_matrix(pcs, mode=0)
   # Plot as heatmap
   ```

5. **Analyze explained variance consistency:**
   - Plot bar chart: explained variance for EOF 1-3 across members
   - Calculate mean and standard deviation

6. **Repeat for EOF modes 2 and 3**

**Key Questions to Answer:**
- Are EOF patterns consistent across ensemble members?
- Which modes are most stable across members?
- Is internal variability affecting lower-order modes?

**Deliverable:**
- Spatial pattern comparison plots (3 panels × N members)
- PC time series comparison plots
- Correlation matrix (members vs members) for each EOF mode
- Summary table of pattern correlations

---

### Day 5-7: Calculate Ensemble Mean EOF

**Goal:** Create an ensemble mean of the first EOF across all members

**Steps:**

1. **Align EOF signs across members:**
   ```python
   # EOFs can have arbitrary sign. Need to align them.
   # Use reference member (e.g., first member)
   reference_eof = eof_patterns[members[0]][0]
   
   for member in members[1:]:
       # Calculate spatial correlation
       corr = correlate_2d_fields(reference_eof, eof_patterns[member][0])
       
       # If negative correlation, flip sign
       if corr < 0:
           eof_patterns[member][0] *= -1
           pcs[member][0] *= -1
   ```

2. **Calculate ensemble mean EOF pattern:**
   ```python
   # For EOF mode 1
   eof_ensemble_mean = np.mean([eof_patterns[m][0] for m in members], axis=0)
   
   # Calculate uncertainty (standard deviation)
   eof_ensemble_std = np.std([eof_patterns[m][0] for m in members], axis=0)
   ```

3. **Calculate ensemble mean PC:**
   ```python
   # Two approaches:
   
   # Approach A: Average PCs
   pc_ensemble_mean = np.mean([pcs[m][0] for m in members], axis=0)
   
   # Approach B: Project data onto ensemble mean EOF
   # (more robust if patterns differ slightly)
   for member in members:
       SST_anom = load_normalized_data(member)
       pc_projected[member] = project_onto_eof(SST_anom, eof_ensemble_mean)
   pc_ensemble_mean = np.mean([pc_projected[m] for m in members], axis=0)
   ```

4. **Visualize ensemble mean:**
   ```python
   fig, axes = plt.subplots(1, 2, figsize=(14, 5))
   
   # Panel 1: Ensemble mean EOF pattern
   plot_map(eof_ensemble_mean, ax=axes[0], title='Ensemble Mean EOF1')
   
   # Panel 2: Uncertainty (std)
   plot_map(eof_ensemble_std, ax=axes[1], title='EOF1 Std Across Members')
   ```

5. **Repeat for EOF modes 2 and 3**

6. **Create ensemble mean for SSS and MLD as well**

**Statistical Analysis:**
- Calculate signal-to-noise ratio: mean/std
- Identify regions of high agreement vs. high uncertainty

**Deliverable:**
- Ensemble mean EOF patterns (modes 1-3) for SST, SSS, MLD
- Ensemble mean PCs (time series)
- Uncertainty maps (standard deviation across members)
- Signal-to-noise ratio maps

---

## Week 2: Member Anomalies & Temporal Analysis

### Day 8-9: Calculate and Plot Member Anomalies

**Goal:** Create plots showing each member minus the ensemble mean

**Steps:**

1. **Calculate anomalies from ensemble mean (EOF patterns):**
   ```python
   for member in members:
       # For EOF mode 1
       eof_anomaly[member] = eof_patterns[member][0] - eof_ensemble_mean
       
       # Calculate RMSE
       rmse[member] = np.sqrt(np.mean(eof_anomaly[member]**2))
   ```

2. **Create comprehensive comparison figure:**
   ```python
   # Figure with N rows (one per member) and 3 columns
   # Column 1: Individual member EOF1
   # Column 2: Ensemble mean EOF1 (repeated)
   # Column 3: Difference (member - mean)
   
   fig, axes = plt.subplots(n_members, 3, figsize=(15, 4*n_members))
   
   for i, member in enumerate(members):
       plot_map(eof_patterns[member][0], ax=axes[i,0], 
                title=f'{member} EOF1')
       plot_map(eof_ensemble_mean, ax=axes[i,1], 
                title='Ensemble Mean EOF1')
       plot_map(eof_anomaly[member], ax=axes[i,2], 
                title=f'{member} - Mean')
   ```

3. **Statistical summary of anomalies:**
   ```python
   # Create box plots
   fig, ax = plt.subplots(figsize=(10, 6))
   data = [eof_anomaly[m].flatten() for m in members]
   ax.boxplot(data, labels=members)
   ax.set_ylabel('EOF1 Anomaly')
   ax.set_title('Distribution of EOF Pattern Anomalies')
   ```

4. **Repeat for PC anomalies:**
   ```python
   for member in members:
       pc_anomaly[member] = pcs[member][0] - pc_ensemble_mean
       
   # Plot all PC anomalies
   plt.figure(figsize=(12, 6))
   for member in members:
       plt.plot(time, pc_anomaly[member], label=member, alpha=0.7)
   plt.axhline(0, color='k', linestyle='--')
   plt.legend()
   plt.title('PC1 Anomalies from Ensemble Mean')
   ```

5. **Quantify inter-member spread:**
   - Calculate correlation between each member and ensemble mean
   - Identify outlier members (if any)
   - Calculate temporal evolution of spread (is it constant or changing?)

**Key Insights to Extract:**
- Which members deviate most from ensemble mean?
- Are deviations systematic or random?
- Do deviations have spatial patterns?
- How does inter-member spread change over time?

**Deliverable:**
- Multi-panel comparison figures (member vs mean)
- Anomaly distribution plots
- PC anomaly time series
- Table: Member | Pattern_Correlation | RMSE | Explained_Variance

---

### Day 10-12: Lagged Correlation Analysis

**Goal:** Investigate temporal relationships and potential predictability

**Steps:**

1. **Load AMOC data for your model:**
   ```python
   # For each member
   for member in members:
       AMOC[member] = load_amoc_data(model, member, years='1840-2014')
       # Calculate annual mean if monthly
       AMOC_annual[member] = AMOC[member].groupby('time.year').mean()
   ```

2. **Calculate lagged correlations between PCs and AMOC:**
   ```python
   lags = np.arange(-20, 21)  # -20 to +20 years
   
   for member in members:
       for mode in [0, 1, 2]:  # EOF 1, 2, 3
           for lag in lags:
               corr = lagged_correlation(pcs[member][mode], 
                                        AMOC_annual[member], 
                                        lag=lag)
               lagged_corr[member][mode][lag] = corr
   
   # Plot lagged correlation
   plt.figure(figsize=(10, 6))
   for member in members:
       plt.plot(lags, lagged_corr[member][0], label=member)
   plt.axhline(0, color='k', linestyle='--')
   plt.xlabel('Lag (years)')
   plt.ylabel('Correlation')
   plt.title('PC1-AMOC Lagged Correlation')
   plt.legend()
   ```

3. **Ensemble mean lagged correlation:**
   ```python
   lagged_corr_ensemble = np.mean([lagged_corr[m][0] for m in members], axis=0)
   lagged_corr_std = np.std([lagged_corr[m][0] for m in members], axis=0)
   
   # Plot with uncertainty
   plt.fill_between(lags, 
                    lagged_corr_ensemble - lagged_corr_std,
                    lagged_corr_ensemble + lagged_corr_std,
                    alpha=0.3)
   plt.plot(lags, lagged_corr_ensemble, linewidth=2)
   ```

4. **Cross-correlation between PC modes:**
   ```python
   # Do PC1 and PC2 have temporal relationships?
   for member in members:
       cross_corr = lagged_correlation(pcs[member][0], pcs[member][1], lags)
   ```

5. **Identify optimal lag times:**
   - At what lag is correlation maximum?
   - Is there predictability (positive lag correlation)?
   - Are there precursor signals?

6. **Test for autocorrelation in PCs:**
   ```python
   # Calculate AR(1) coefficient
   for member in members:
       ar1[member] = calculate_ar1(pcs[member][0])
   
   # Increasing AR(1) could indicate critical slowing down
   # Plot AR(1) in sliding windows
   window = 30  # years
   ar1_timeseries = sliding_window_ar1(pcs[member][0], window)
   ```

**Advanced Analysis (if time permits):**
- Wavelet analysis to see frequency-dependent correlations
- Granger causality tests
- Lead-lag relationships between SST and SSS EOF modes

**Deliverable:**
- Lagged correlation plots (PC vs AMOC) for each member
- Ensemble mean lagged correlation with uncertainty bands
- Cross-correlation matrices between different PC modes
- AR(1) time series in sliding windows
- Summary: optimal lag times and predictability estimates

---

### Day 13-14: Document Findings & Prepare Summary

**Goal:** Synthesize Week 1-2 results and prepare for next phase

**Steps:**

1. **Create summary document with key findings:**
   - Consistency of EOF patterns across members
   - Ensemble mean characteristics
   - Inter-member variability
   - Temporal relationships (lagged correlations)

2. **Prepare comparison table:**
   ```
   | Member | EOF1 Var | EOF1-Mean Corr | PC1-AMOC Corr | Optimal Lag |
   |--------|----------|----------------|---------------|-------------|
   | r1i1p1 | 25.3%    | 0.95          | -0.42         | -5 years    |
   | r2i1p1 | 24.8%    | 0.93          | -0.38         | -4 years    |
   | ...    | ...      | ...           | ...           | ...         |
   ```

3. **Identify key patterns for presentation:**
   - Select best figures for supervisor meeting
   - Highlight interesting findings (e.g., predictability at certain lags)

4. **Note questions and issues:**
   - Any problematic members?
   - Unexpected patterns that need explanation?
   - Technical challenges encountered?

**Deliverable:**
- Summary document (2-3 pages)
- Selection of best figures (5-7 key plots)
- List of questions for supervisors

---

## Week 3: Comparison with Observations & Planning 3D Extension

### Day 15-17: Compare with Observed Data (EN4)

**Goal:** Validate model EOF patterns against observations

**Steps:**

1. **Load and prepare observational data:**
   ```python
   # EN4 dataset
   obs_sst = load_EN4_data('temperature', depth='surface', years='1840-2014')
   obs_sss = load_EN4_data('salinity', depth='surface', years='1840-2014')
   
   # Note: EN4 may not go back to 1840, typically starts ~1900
   # Adjust time period to overlap: e.g., 1950-2014
   overlap_period = '1950-2014'
   ```

2. **Regrid if necessary:**
   ```python
   # Ensure obs and model are on same grid
   obs_sst_regridded = regrid_to_model_grid(obs_sst, model_grid)
   ```

3. **Perform EOF analysis on observations:**
   ```python
   # Same procedure as for model
   obs_sst_anom = obs_sst - obs_sst.groupby('time.month').mean('time')
   obs_eof_patterns, obs_pcs, obs_explained_var = calculate_eof(obs_sst_anom, n_modes=10)
   ```

4. **Compare EOF patterns: Model vs Observations:**
   ```python
   # For EOF mode 1
   pattern_corr = correlate_2d_fields(eof_ensemble_mean, obs_eof_patterns[0])
   
   # Create comparison figure
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   plot_map(eof_ensemble_mean, ax=axes[0], title='Model Ensemble Mean EOF1')
   plot_map(obs_eof_patterns[0], ax=axes[1], title='Observed EOF1')
   plot_map(eof_ensemble_mean - obs_eof_patterns[0], ax=axes[2], 
            title='Difference')
   
   axes[0].text(0.05, 0.95, f'Corr: {pattern_corr:.2f}', 
                transform=axes[0].transAxes)
   ```

5. **Compare PC time series:**
   ```python
   # Normalize both PCs for comparison
   model_pc1_norm = normalize(pc_ensemble_mean)
   obs_pc1_norm = normalize(obs_pcs[0])
   
   plt.figure(figsize=(12, 6))
   plt.plot(time, model_pc1_norm, label='Model Ensemble Mean', linewidth=2)
   plt.plot(time, obs_pc1_norm, label='Observations (EN4)', linewidth=2)
   plt.legend()
   plt.title('PC1 Comparison: Model vs Observations')
   
   # Calculate correlation
   pc_corr = np.corrcoef(model_pc1_norm, obs_pc1_norm)[0,1]
   ```

6. **Assess model performance:**
   - Spatial pattern correlation
   - Explained variance comparison
   - Temporal correlation of PCs
   - Amplitude comparison

7. **Repeat for SSS if observational data quality is sufficient**

8. **Additional observational datasets (if time):**
   - HadISST for SST
   - IAP for subsurface temperature
   - Ishii for salinity

**Key Questions:**
- Does the model capture observed EOF patterns?
- Are there systematic biases?
- Which model members best match observations?
- Can we constrain model uncertainty using observations?

**Deliverable:**
- Model vs Observations comparison figures (EOF patterns)
- Model vs Observations PC time series comparison
- Correlation and skill metrics table
- Assessment of model fidelity

---

### Day 18-19: Plan 3D Extension

**Goal:** Design strategy for extending EOF analysis to 3D ocean fields

**Steps:**

1. **Review data requirements:**
   ```python
   # 3D temperature and salinity
   # Variables: thetao (3D potential temperature), so (3D salinity)
   
   # Consider data volume:
   # - Spatial dimensions: lat × lon × depth
   # - Temporal: 1840-2014 (175 years)
   # - Multiple members
   # This is LARGE data!
   ```

2. **Design 3D EOF approach:**

   **Option A: Depth-integrated EOFs**
   ```python
   # Integrate over specific depth ranges
   temp_0_300m = vertical_integral(temp_3d, depth_range=[0, 300])
   temp_300_1000m = vertical_integral(temp_3d, depth_range=[300, 1000])
   # Then perform EOF on integrated fields
   ```

   **Option B: Depth-level EOFs**
   ```python
   # Perform EOF separately at key depth levels
   depths_of_interest = [0, 100, 300, 500, 1000, 2000]
   for depth in depths_of_interest:
       eof_at_depth = calculate_eof(temp_3d.sel(depth=depth))
   ```

   **Option C: Full 3D EOF**
   ```python
   # Reshape 3D data to 2D: (time, space×depth)
   temp_reshaped = temp_3d.reshape(time, lat*lon*depth)
   # Perform EOF on reshaped data
   # Patterns will be 3D structures
   ```

   **Option D: Vertical modes first, then horizontal**
   ```python
   # First decompose vertically (at each horizontal point)
   # Then perform horizontal EOF on vertical mode amplitudes
   # More physically interpretable
   ```

3. **Assess computational feasibility:**
   - Estimate memory requirements
   - Test with single member first
   - Consider downsampling strategies:
     * Spatial coarsening
     * Temporal subsampling (e.g., annual means only)
     * Depth level selection

4. **Select focus region:**
   ```python
   # To reduce data volume, focus on AMOC-relevant region
   # North Atlantic: 80°W-0°W, 0°N-70°N
   temp_3d_atlantic = temp_3d.sel(lon=slice(-80, 0), lat=slice(0, 70))
   ```

5. **Create test implementation:**
   ```python
   # Simple test with one member, one year
   test_data = load_3d_data(model, member='r1i1p1f1', year=2000)
   
   # Try each approach and time it
   import time
   start = time.time()
   eof_3d = calculate_3d_eof(test_data)
   elapsed = time.time() - start
   print(f"3D EOF took {elapsed:.1f} seconds")
   ```

6. **Design analysis pipeline:**
   ```
   1. Load 3D data (by chunks if needed)
   2. Preprocess (remove climatology, detrend if needed)
   3. Apply EOF method (chosen from options above)
   4. Validate reconstruction quality
   5. Relate to AMOC
   6. Compare across members
   ```

7. **Identify challenges and solutions:**
   ```
   Challenge 1: Data volume too large
   → Solution: Use dask for lazy loading, process in chunks
   
   Challenge 2: EOF computation slow
   → Solution: Use randomized SVD for large matrices
   
   Challenge 3: Interpretation of 3D patterns
   → Solution: Create cross-sections and vertical profiles
   ```

**Deliverable:**
- Document outlining 3D EOF strategy
- Pros/cons of each approach
- Test results with single member
- Computational requirements estimate
- Recommended approach with justification
- Timeline for 3D implementation (Week 4+)

---

### Day 20-21: Explore Neural Network Approach

**Goal:** Investigate using neural networks to learn the latent space

**Steps:**

1. **Research autoencoder architectures for climate data:**
   - Convolutional autoencoders for spatial data
   - Variational autoencoders (VAE) for uncertainty quantification
   - Recent papers using neural networks for climate EOF-like analysis

2. **Design basic autoencoder architecture:**
   ```python
   import tensorflow as tf
   from tensorflow import keras
   
   # Simple autoencoder for SST fields
   input_shape = (lat_dim, lon_dim)
   latent_dim = 10  # Compare to 10 EOF modes
   
   # Encoder
   encoder = keras.Sequential([
       keras.layers.Input(shape=input_shape),
       keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
       keras.layers.MaxPooling2D(2),
       keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
       keras.layers.MaxPooling2D(2),
       keras.layers.Flatten(),
       keras.layers.Dense(latent_dim)  # Latent space
   ])
   
   # Decoder
   decoder = keras.Sequential([
       keras.layers.Input(shape=(latent_dim,)),
       keras.layers.Dense(reduced_spatial_dim),
       keras.layers.Reshape((reduced_lat, reduced_lon, 64)),
       keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same'),
       keras.layers.UpSampling2D(2),
       keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same'),
       keras.layers.UpSampling2D(2),
       keras.layers.Conv2D(1, 3, activation='linear', padding='same')
   ])
   
   autoencoder = keras.Model(inputs=encoder.input, 
                            outputs=decoder(encoder.output))
   ```

3. **Prepare training data:**
   ```python
   # Use your model ensemble data
   # Split: 80% training, 20% validation
   
   X_train = []
   for member in members:
       SST_anom = load_normalized_data(member, 'SST', years='1840-2000')
       X_train.append(SST_anom)
   X_train = np.concatenate(X_train)
   
   X_val = [load for years 2000-2014]
   ```

4. **Train autoencoder:**
   ```python
   autoencoder.compile(optimizer='adam', loss='mse')
   history = autoencoder.fit(X_train, X_train,  # Reconstruct input
                            validation_data=(X_val, X_val),
                            epochs=100,
                            batch_size=32)
   ```

5. **Compare with EOF results:**
   ```python
   # Reconstruction quality
   sst_reconstructed_ae = autoencoder.predict(X_test)
   sst_reconstructed_eof = reconstruct_from_eof(X_test, eof_patterns, n_modes=10)
   
   # Compare RMSE
   rmse_ae = np.sqrt(np.mean((X_test - sst_reconstructed_ae)**2))
   rmse_eof = np.sqrt(np.mean((X_test - sst_reconstructed_eof)**2))
   
   print(f"Autoencoder RMSE: {rmse_ae:.4f}")
   print(f"EOF RMSE: {rmse_eof:.4f}")
   ```

6. **Analyze learned latent space:**
   ```python
   # Extract latent representations
   latent_representations = encoder.predict(X_train)
   
   # Do they correlate with AMOC?
   for i in range(latent_dim):
       corr = correlate(latent_representations[:, i], AMOC)
       print(f"Latent dimension {i} - AMOC correlation: {corr:.3f}")
   ```

7. **Compare latent space: Neural network vs EOF:**
   - Are neural network latent variables similar to PCs?
   - Does neural network capture nonlinear relationships?
   - Is reconstruction better with same dimensionality?

8. **Document findings and next steps:**
   - Advantages of neural network approach
   - Computational requirements
   - Interpretability vs performance trade-off
   - Recommendations for full implementation

**Literature to Review:**
- Ham et al. (2019) - Deep learning for ENSO prediction
- Sonnewald et al. (2021) - Unsupervised learning for ocean dynamics
- Toms et al. (2020) - ML for climate extremes
- Kadow et al. (2020) - Artificial intelligence reconstructs missing climate information

**Deliverable:**
- Literature review summary (1-2 pages)
- Basic autoencoder implementation and results
- Comparison: EOF vs Neural Network (table with metrics)
- Recommendations for which approach to pursue
- Timeline for full neural network implementation

---

## End of Week 3: Summary & Next Steps Planning

### Comprehensive Summary Document

Create a summary report covering:

#### **1. Multi-Member EOF Analysis Results**
- Number of members analyzed
- Consistency of EOF patterns across members
- Explained variance statistics
- Ensemble mean characteristics
- Inter-member variability quantification

#### **2. Temporal Analysis Results**
- Lagged correlation findings
- PC-AMOC relationships
- Optimal lag times for prediction
- Autocorrelation and critical slowing down indicators

#### **3. Observational Validation**
- Model-observation comparison
- Pattern and temporal correlations
- Model bias assessment
- Ensemble member ranking (which match observations best)

#### **4. Future Directions**
- 3D EOF analysis plan
- Neural network approach feasibility
- Timeline for implementation

### Key Deliverables Checklist

**✅ Data Products:**
- [ ] EOF patterns for all members (SST, SSS, MLD)
- [ ] Principal components for all members
- [ ] Ensemble mean EOFs and PCs
- [ ] Member anomalies from ensemble mean
- [ ] Lagged correlation results
- [ ] Observed EOF patterns from EN4
- [ ] AMOC data for all members

**✅ Figures (15-20 total):**
1. [ ] Spatial patterns: EOF 1-3 for all members (SST, SSS, MLD)
2. [ ] PC time series comparison across members
3. [ ] Ensemble mean EOF patterns with uncertainty
4. [ ] Member anomaly plots (member minus mean)
5. [ ] Lagged correlation plots (PC-AMOC)
6. [ ] AR(1) in sliding windows
7. [ ] Model vs Observation EOF comparison
8. [ ] Model vs Observation PC comparison
9. [ ] 3D EOF test results (if completed)
10. [ ] Neural network reconstruction comparison (if completed)

**✅ Tables:**
- [ ] Member statistics (explained variance, correlations)
- [ ] Pattern correlation matrix (members vs members)
- [ ] Lagged correlation optimal lags
- [ ] Model vs Observation metrics
- [ ] Reconstruction quality comparison (EOF vs NN)

**✅ Written Documents:**
- [ ] Week 1-3 summary (5-10 pages)
- [ ] 3D EOF strategy document
- [ ] Neural network feasibility assessment
- [ ] Questions for supervisors
- [ ] Timeline for next phase (Weeks 4-8)

---

## Practical Implementation Tips

### Managing Multiple Members

```python
# Efficient data management
import xarray as xr
import dask

# Load multiple members at once
ds_all_members = xr.open_mfdataset(
    f'{data_path}/tos_{model}_r*i1p1f1_*.nc',
    combine='nested',
    concat_dim='member',
    parallel=True,
    chunks={'time': 120}  # Chunk for efficient processing
)
```

### Saving Results Systematically

```python
# Organize output directory
output_structure = """
results/
├── eof_patterns/
│   ├── sst/
│   │   ├── r1i1p1f1_eof_patterns.nc
│   │   ├── r2i1p1f1_eof_patterns.nc
│   │   └── ensemble_mean_eof.nc
│   ├── sss/
│   └── mld/
├── pcs/
│   ├── sst/
│   │   ├── r1i1p1f1_pcs.nc
│   │   └── ensemble_mean_pc.nc
│   ├── sss/
│   └── mld/
├── lagged_correlations/
│   ├── pc_amoc_lagged_corr.nc
│   └── cross_mode_correlations.nc
├── observations/
│   ├── en4_eof_patterns.nc
│   └── model_obs_comparison.nc
└── figures/
    ├── week1/
    ├── week2/
    └── week3/
"""

# Save function
def save_eof_results(member, variable, eof_patterns, pcs, explained_var):
    """Systematically save EOF analysis results"""
    output_dir = f'results/eof_patterns/{variable}/'
    
    # Save as NetCDF with metadata
    ds = xr.Dataset({
        'eof_patterns': (['mode', 'lat', 'lon'], eof_patterns),
        'explained_variance': (['mode'], explained_var),
    })
    ds.attrs['member'] = member
    ds.attrs['variable'] = variable
    ds.attrs['time_period'] = '1840-2014'
    ds.attrs['n_modes'] = len(explained_var)
    
    ds.to_netcdf(f'{output_dir}/{member}_eof_patterns.nc')
    
    # Save PCs separately
    pc_ds = xr.Dataset({
        'pc': (['mode', 'time'], pcs),
    })
    pc_ds.to_netcdf(f'results/pcs/{variable}/{member}_pcs.nc')
```

### Version Control & Reproducibility

```python
# Keep a log of processing
import json
from datetime import datetime

processing_log = {
    'date': datetime.now().isoformat(),
    'model': 'CESM2',
    'members': ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1'],
    'time_period': '1840-2014',
    'variables': ['SST', 'SSS', 'MLD'],
    'n_eof_modes': 10,
    'preprocessing': {
        'deseasonalized': True,
        'detrended': False,
        'normalization': 'standardized'
    },
    'notes': 'Initial multi-member EOF analysis'
}

with open('results/processing_log.json', 'w') as f:
    json.dump(processing_log, f, indent=2)
```

### Efficient EOF Calculation for Large Data

```python
# For large datasets, use dask-ml or scikit-learn with sparse methods
from sklearn.decomposition import IncrementalPCA
import numpy as np

def calculate_eof_efficient(data, n_modes=10):
    """
    Calculate EOF using incremental PCA for large datasets
    
    Parameters:
    -----------
    data : xarray.DataArray
        Shape (time, lat, lon)
    n_modes : int
        Number of EOF modes to calculate
    
    Returns:
    --------
    eof_patterns : array (n_modes, lat, lon)
    pcs : array (n_modes, time)
    explained_var : array (n_modes,)
    """
    # Reshape to (time, space)
    nt = data.shape[0]
    spatial_shape = data.shape[1:]
    data_2d = data.values.reshape(nt, -1)
    
    # Remove NaN (land points)
    valid_mask = ~np.isnan(data_2d[0, :])
    data_2d_valid = data_2d[:, valid_mask]
    
    # Standardize
    data_mean = np.nanmean(data_2d_valid, axis=0)
    data_std = np.nanstd(data_2d_valid, axis=0)
    data_normalized = (data_2d_valid - data_mean) / data_std
    
    # Calculate EOFs using PCA
    pca = IncrementalPCA(n_components=n_modes)
    pcs = pca.fit_transform(data_normalized)  # Shape: (time, n_modes)
    
    # EOF patterns are the components
    eof_spatial = pca.components_  # Shape: (n_modes, space)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    
    # Reshape EOFs back to spatial grid
    eof_patterns = np.full((n_modes, *spatial_shape), np.nan)
    for mode in range(n_modes):
        eof_full = np.full(np.prod(spatial_shape), np.nan)
        eof_full[valid_mask] = eof_spatial[mode, :]
        eof_patterns[mode] = eof_full.reshape(spatial_shape)
    
    return eof_patterns, pcs.T, explained_var  # Return (modes, time) for PCs
```

---

## Troubleshooting Guide

### Problem: Ensemble members have different time periods

**Solution:**
```python
# Find common time period
def find_common_period(members):
    time_ranges = {}
    for member in members:
        ds = xr.open_dataset(f'{data_path}/{member}_tos.nc')
        time_ranges[member] = (ds.time.min().values, ds.time.max().values)
    
    # Find overlap
    start_times = [t[0] for t in time_ranges.values()]
    end_times = [t[1] for t in time_ranges.values()]
    
    common_start = max(start_times)
    common_end = min(end_times)
    
    print(f"Common period: {common_start} to {common_end}")
    return common_start, common_end

# Use common period for all members
common_start, common_end = find_common_period(members)
for member in members:
    data = load_data(member).sel(time=slice(common_start, common_end))
```

### Problem: EOF signs are inconsistent across members

**Solution:**
```python
def align_eof_signs(eof_patterns, reference_eof):
    """
    Align EOF signs to a reference pattern
    
    Parameters:
    -----------
    eof_patterns : dict
        Dictionary of EOF patterns {member: patterns}
    reference_eof : array
        Reference EOF pattern (e.g., from first member)
    
    Returns:
    --------
    aligned_patterns : dict
        EOFs with consistent signs
    flip_flags : dict
        Which members were flipped
    """
    aligned_patterns = {}
    flip_flags = {}
    
    for member, pattern in eof_patterns.items():
        # Calculate spatial correlation with reference
        # Ignore NaN values (land points)
        valid = ~(np.isnan(pattern) | np.isnan(reference_eof))
        corr = np.corrcoef(pattern[valid], reference_eof[valid])[0, 1]
        
        if corr < 0:
            aligned_patterns[member] = -pattern
            flip_flags[member] = True
        else:
            aligned_patterns[member] = pattern
            flip_flags[member] = False
    
    return aligned_patterns, flip_flags
```

### Problem: Lagged correlation analysis is slow

**Solution:**
```python
import scipy.signal as signal

def fast_lagged_correlation(x, y, max_lag=20):
    """
    Fast lagged correlation using FFT
    
    Parameters:
    -----------
    x, y : array
        Time series to correlate
    max_lag : int
        Maximum lag in time steps
    
    Returns:
    --------
    lags : array
        Lag values
    correlations : array
        Correlation at each lag
    """
    # Normalize
    x_norm = (x - np.mean(x)) / np.std(x)
    y_norm = (y - np.mean(y)) / np.std(y)
    
    # Use scipy's correlate (faster than manual loops)
    corr = signal.correlate(y_norm, x_norm, mode='full') / len(x)
    
    # Extract relevant lags
    lags = signal.correlation_lags(len(x), len(y), mode='full')
    mask = (lags >= -max_lag) & (lags <= max_lag)
    
    return lags[mask], corr[mask]
```

### Problem: EN4 observational data has different resolution than model

**Solution:**
```python
def regrid_to_common_grid(model_data, obs_data, target_resolution=1.0):
    """
    Regrid both model and observations to common grid
    
    Parameters:
    -----------
    model_data : xarray.DataArray
    obs_data : xarray.DataArray
    target_resolution : float
        Target grid resolution in degrees
    
    Returns:
    --------
    model_regridded, obs_regridded : xarray.DataArray
    """
    import xesmf as xe
    
    # Define target grid
    target_grid = xr.Dataset({
        'lat': np.arange(-90, 90, target_resolution),
        'lon': np.arange(0, 360, target_resolution)
    })
    
    # Create regridders
    regridder_model = xe.Regridder(model_data, target_grid, 'bilinear')
    regridder_obs = xe.Regridder(obs_data, target_grid, 'bilinear')
    
    # Regrid
    model_regridded = regridder_model(model_data)
    obs_regridded = regridder_obs(obs_data)
    
    return model_regridded, obs_regridded
```

### Problem: Memory issues with multiple members

**Solution:**
```python
# Process members sequentially, not all at once
def process_members_sequentially(members, variables):
    """
    Process members one at a time to avoid memory issues
    """
    results = {}
    
    for member in members:
        print(f"Processing {member}...")
        
        # Load only this member
        data = load_member_data(member, variables)
        
        # Calculate EOF
        eof_result = calculate_eof(data)
        
        # Save immediately
        save_eof_results(member, eof_result)
        
        # Store only essential info, not full data
        results[member] = {
            'explained_var': eof_result['explained_var'],
            'pattern_shape': eof_result['patterns'].shape
        }
        
        # Clear memory
        del data, eof_result
        import gc
        gc.collect()
    
    return results

# Then load saved results for comparison
def load_saved_eof_patterns(members):
    """Load previously saved EOF patterns"""
    patterns = {}
    for member in members:
        ds = xr.open_dataset(f'results/eof_patterns/sst/{member}_eof_patterns.nc')
        patterns[member] = ds['eof_patterns'].values
    return patterns
```

---

## Weekly Progress Tracking Template

### Week 1 Progress Tracker

**Monday-Tuesday:**
- [ ] Downloaded data for all members
- [ ] Verified data completeness (1840-2014)
- [ ] Performed EOF on member 1 (SST)
- [ ] Performed EOF on member 2 (SST)
- [ ] Performed EOF on member 3 (SST)
- [ ] ...
- [ ] Documented any data issues

**Wednesday-Thursday:**
- [ ] Created comparison plots for EOF modes 1-3
- [ ] Calculated pattern correlations between members
- [ ] Plotted PC time series for all members
- [ ] Analyzed explained variance consistency
- [ ] Identified any outlier members

**Friday-Sunday:**
- [ ] Aligned EOF signs across members
- [ ] Calculated ensemble mean EOF patterns
- [ ] Calculated ensemble mean PCs
- [ ] Created uncertainty maps (std across members)
- [ ] Completed Week 1 summary document

### Week 2 Progress Tracker

**Monday-Tuesday:**
- [ ] Calculated EOF anomalies (member - mean)
- [ ] Created multi-panel comparison figures
- [ ] Quantified inter-member spread
- [ ] Analyzed spatial patterns of anomalies

**Wednesday-Friday:**
- [ ] Loaded AMOC data for all members
- [ ] Calculated lagged correlations (PC-AMOC)
- [ ] Created lagged correlation plots
- [ ] Calculated ensemble mean lagged correlations
- [ ] Computed AR(1) in sliding windows
- [ ] Analyzed cross-correlations between modes

**Saturday-Sunday:**
- [ ] Synthesized Week 1-2 findings
- [ ] Created summary tables and figures
- [ ] Prepared questions for supervisors

### Week 3 Progress Tracker

**Monday-Wednesday:**
- [ ] Downloaded EN4 observational data
- [ ] Performed EOF on observations
- [ ] Regridded model and obs to common grid
- [ ] Compared EOF patterns (model vs obs)
- [ ] Compared PC time series
- [ ] Calculated skill metrics
- [ ] Assessed model biases

**Thursday-Friday:**
- [ ] Researched 3D EOF approaches
- [ ] Tested computational requirements
- [ ] Designed 3D analysis pipeline
- [ ] Created 3D strategy document

**Saturday-Sunday:**
- [ ] Reviewed neural network literature
- [ ] Implemented basic autoencoder
- [ ] Trained on ensemble data
- [ ] Compared with EOF results
- [ ] Documented NN feasibility

---

## Questions for Supervisors (Week 3 Meeting)

### About Current Results:

1. **Multi-member consistency:**
   - "I find X% pattern correlation between members for EOF1. Is this sufficient consistency?"
   - "Should I exclude any members that deviate significantly from ensemble mean?"

2. **Lagged correlations:**
   - "PC1 shows maximum correlation with AMOC at lag of -X years. How do I interpret this?"
   - "AR(1) shows increasing trend in later period. Could this indicate critical slowing down?"

3. **Observational comparison:**
   - "Model EOF1 correlates at r=X with EN4. Is this good agreement?"
   - "Model overestimates variance in region Y. Should I bias-correct before further analysis?"

### About Future Direction:

4. **3D Extension:**
   - "Which 3D EOF approach do you recommend: depth-integrated, level-by-level, or full 3D?"
   - "Should I focus on Atlantic basin only or global ocean?"
   - "What depth range is most relevant for AMOC: 0-1000m, 0-3000m, full depth?"

5. **Neural Networks:**
   - "Should I pursue neural network approach now or complete linear EOF analysis first?"
   - "Is interpretability or reconstruction accuracy more important for this thesis?"
   - "Do you have computational resources available for training neural networks?"

6. **Scope & Timeline:**
   - "For the remaining thesis time, should I prioritize:
     * Depth (3D analysis)?
     * Breadth (more models)?
     * Methods (neural networks)?
     * Applications (early warning signals)?"

7. **Early Warning Signals:**
   - "When should I start implementing critical slowing down detection?"
   - "Should I test early warning indicators on ensemble mean or individual members?"

8. **Thesis Structure:**
   - "How many chapters should the thesis have?"
   - "Should validation and application be separate chapters?"
   - "Do you want preliminary results for [upcoming seminar/conference]?"

---

## Next Steps After Week 3

### Short-term (Weeks 4-6): Choose One Primary Focus

**Option A: 3D EOF Analysis**
- Extend EOF to subsurface ocean
- Analyze vertical structure of AMOC-related patterns
- Compare surface vs subsurface early warning signals

**Option B: Multi-Model Analysis**
- Apply EOF framework to multiple CMIP6 models
- Compare AMOC-EOF relationships across models
- Identify robust vs model-dependent features

**Option C: Early Warning Signal Detection**
- Apply critical slowing down metrics to existing PCs
- Test various indicators (AR1, variance, DFA, etc.)
- Validate against known AMOC changes in models

**Option D: Neural Network Implementation**
- Develop and train full autoencoder
- Compare linear (EOF) vs nonlinear (NN) latent spaces
- Test if NN captures additional AMOC-relevant information

### Medium-term (Weeks 7-10): Integration & Application

- Combine best approaches from weeks 4-6
- Apply to future climate scenarios (SSP projections)
- Test predictability of AMOC changes
- Compare with published early warning studies

### Long-term (Weeks 11-15): Thesis Writing

- Introduction & literature review
- Methods chapter
- Results chapters (2-3 chapters)
- Discussion & conclusions
- Revisions and defense preparation

---

## Success Criteria - End of Week 3

### You should be able to answer these questions:

**Technical Questions:**
1. ✅ How many EOF modes are needed to capture X% of variance?
2. ✅ Are EOF patterns consistent across ensemble members?
3. ✅ What is the inter-member spread in EOF1?
4. ✅ Do PCs correlate with AMOC strength?
5. ✅ At what lag is PC-AMOC correlation strongest?
6. ✅ Do model EOFs match observed EOFs?
7. ✅ What is model bias in EOF patterns?

**Conceptual Questions:**
1. ✅ What does EOF1 represent physically?
2. ✅ Why do members differ in their EOF patterns?
3. ✅ What causes lagged correlations between PCs and AMOC?
4. ✅ How does internal variability affect EOF analysis?
5. ✅ Can surface EOF patterns predict subsurface AMOC changes?

**Practical Questions:**
1. ✅ Which approach for 3D analysis is most feasible?
2. ✅ Should I use neural networks or stick with EOF?
3. ✅ What is the computational cost of each approach?
4. ✅ What are the next priority tasks?

### Expected Achievements:

**Completed:**
- ✅ Multi-member EOF analysis (5-10 members)
- ✅ Ensemble mean EOF calculation
- ✅ Member anomaly quantification
- ✅ Lagged correlation analysis
- ✅ Model-observation comparison
- ✅ 3D and NN feasibility assessment

**Deliverables:**
- ✅ 15-20 publication-quality figures
- ✅ 5-8 summary tables
- ✅ Week 3 comprehensive report (5-10 pages)
- ✅ Strategy documents for next phase
- ✅ Code repository with documented functions

**Understanding:**
- ✅ Strengths and limitations of EOF approach
- ✅ Relationship between EOF patterns and AMOC
- ✅ Role of internal variability in EOF analysis
- ✅ Model fidelity in representing observations
- ✅ Feasibility of different analysis paths forward

---

## Resources & References

### Key Papers for EOF Analysis:

1. **Hannachi et al. (2007)** - "Empirical orthogonal functions and related techniques in atmospheric science"
   - Comprehensive review of EOF methods

2. **North et al. (1982)** - "Sampling errors in the estimation of empirical orthogonal functions"
   - Understanding EOF uncertainty

3. **Monahan et al. (2009)** - "Empirical Orthogonal Functions: The Medium is the Message"
   - Interpretation and physical meaning of EOFs

### For AMOC Analysis:

4. **Caesar et al. (2021)** - "Current Atlantic Meridional Overturning Circulation weakest in last millennium"
   - Context for AMOC changes

5. **Jackson et al. (2022)** - "The evolution of the North Atlantic Meridional Overturning Circulation since 1980"
   - Recent AMOC trends in models and observations

6. **Boers (2021)** - "Observation-based early-warning signals for a collapse of the AMOC"
   - Early warning signal methodology

### For Neural Networks in Climate:

7. **Reichstein et al. (2019)** - "Deep learning and process understanding for data-driven Earth system science"
   - Overview of ML in climate science

8. **Ham et al. (2019)** - "Deep learning for multi-year ENSO forecasts"
   - CNN/LSTM for climate prediction

9. **Sonnewald et al. (2021)** - "Bridging observations, theory and numerical simulation of the ocean using machine learning"
   - Unsupervised learning for ocean dynamics

### Software & Tools:

**Python Packages:**
```python
xarray         # NetCDF data handling
numpy          # Numerical operations
scipy          # Statistical functions
matplotlib     # Plotting
cartopy        # Maps
sklearn        # PCA/EOF
tensorflow     # Neural networks
dask           # Parallel computing
xesmf          # Regridding
eofs           # Dedicated EOF package
```

**Useful Resources:**
- xarray tutorial: https://tutorial.xarray.dev
- CMIP6 data access: https://esgf-node.llnl.gov/
- EN4 data: https://www.metoffice.gov.uk/hadobs/en4/
- EOF package docs: https://ajdawson.github.io/eofs/

---

## Final Checklist: Ready for Week 4?

**Data Preparation:**
- [ ] All ensemble members identified and accessible
- [ ] AMOC data downloaded for all members
- [ ] Observational data (EN4) downloaded and preprocessed
- [ ] Data directory organized systematically

**Analysis Complete:**
- [ ] EOFs calculated for all members (SST, SSS, MLD)
- [ ] Ensemble mean EOFs calculated
- [ ] Member anomalies quantified
- [ ] Lagged correlations computed
- [ ] Model-observation comparison done
- [ ] 3D and NN approaches tested

**Documentation:**
- [ ] All figures saved with descriptive names
- [ ] All results saved in organized directory structure
- [ ] Processing log maintained
- [ ] Code commented and organized
- [ ] Summary report written

**Understanding:**
- [ ] Understand what EOF patterns represent
- [ ] Know which members best match observations
- [ ] Identified optimal lag for PC-AMOC relationship
- [ ] Decided on approach for next phase

**Planning:**
- [ ] Week 4+ timeline created
- [ ] Questions prepared for supervisors
- [ ] Identified potential challenges
- [ ] Backup plans if primary approach fails

---

**Remember:** The goal of these 3 weeks is to establish a solid foundation for your latent space analysis. Take time to understand the results deeply, not just produce figures mechanically. The insights you gain now will guide the rest of your thesis!

**Good luck!** 🚀

---

**Document Version:** 2.0 - Updated Plan (No FPI)  
**Last Updated:** October 28, 2025  
**Next Review:** End of Week 3