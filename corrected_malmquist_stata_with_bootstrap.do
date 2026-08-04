********************************************************************************
* Corrected Malmquist productivity analysis in Stata
* Kenya counties, 2014 to 2022
*
* Point estimator:
*   - DEA Malmquist index from malmq2
*   - output orientation
*   - contemporaneous technology
*   - CRS productivity index with FGNZ decomposition
*
* Optional bootstrap:
*   - resamples complete county panels, not individual county-year rows
*   - uses resampled counties to construct each period frontier
*   - always evaluates the original counties, preserving county identity
*   - reports percentile confidence intervals for the three CRS indices
*
* IMPORTANT INFERENCE NOTE:
*   The optional bootstrap is a paired nonparametric frontier-resampling
*   sensitivity analysis. It is NOT the smooth Simar-Wilson (1999) Malmquist
*   bootstrap. Do not describe these intervals as Simar-Wilson intervals.
********************************************************************************

version 16.0
clear all
set more off
set linesize 255

********************************************************************************
* 1. USER SETTINGS
********************************************************************************

* Edit these three paths.
global PROJECT "G:/My Drive/Kenya Subnational"
global DATA    "${PROJECT}/mydata.csv"
global OUT     "${PROJECT}/stata_results"

* Model specification matching the corrected R analysis.
local MODEL   "combined"
local INPUTS  "pop_dev hw_updated"
local OUTPUTS "childsurvival mumsurvival immunization"

* Other examples:
* local MODEL   "child"
* local OUTPUTS "childsurvival"
*
* local MODEL   "maternal"
* local OUTPUTS "mumsurvival"
*
* local MODEL   "hale"
* local OUTPUTS "hale"

* Bootstrap switch and settings.
local RUN_BOOTSTRAP 1
local BOOT_REPS     1000
local BOOT_SEED     123456789
local CI_LEVEL      95

if !inlist(`RUN_BOOTSTRAP', 0, 1) {
    display as error "RUN_BOOTSTRAP must be 0 or 1."
    exit 198
}
if `BOOT_REPS' < 1 | `BOOT_REPS' != floor(`BOOT_REPS') {
    display as error "BOOT_REPS must be a positive integer."
    exit 198
}
if `CI_LEVEL' <= 0 | `CI_LEVEL' >= 100 {
    display as error "CI_LEVEL must be strictly between 0 and 100."
    exit 198
}

capture mkdir "${OUT}"

********************************************************************************
* 2. INSTALL AND VERIFY MALMQ2
********************************************************************************

capture which malmq2
if _rc {
    display as text "Installing malmq2 from SSC..."
    ssc install malmq2
}

capture which malmq2
if _rc {
    display as error "malmq2 is not available. Install it and rerun the do-file."
    exit 499
}

********************************************************************************
* 3. LOAD AND VALIDATE THE ANALYSIS PANEL
********************************************************************************

capture confirm file "${DATA}"
if _rc {
    display as error "Data file not found: ${DATA}"
    exit 601
}

import delimited using "${DATA}", clear varnames(1) case(preserve)

* Standardize the immunization variable name to lower case.
capture confirm variable immunization
if _rc {
    capture confirm variable Immunization
    if _rc {
        display as error "Neither immunization nor Immunization was found."
        exit 111
    }
    rename Immunization immunization
}

capture confirm variable location_name
if _rc {
    display as error "Required variable location_name was not found."
    exit 111
}

capture confirm variable year
if _rc {
    display as error "Required variable year was not found."
    exit 111
}

capture confirm numeric variable year
if _rc {
    destring year, replace
}

capture confirm string variable location_name
if _rc {
    tostring location_name, replace force
}
replace location_name = strtrim(itrim(location_name))

keep if inlist(year, 2014, 2022)
count
if r(N) == 0 {
    display as error "No observations for 2014 or 2022."
    exit 2000
}

* Check that all required DEA variables exist, are numeric, and are positive.
foreach v of local INPUTS {
    capture confirm variable `v'
    if _rc {
        display as error "Required input variable `v' was not found."
        exit 111
    }
    capture confirm numeric variable `v'
    if _rc {
        destring `v', replace ignore(",")
    }
    assert !missing(`v')
    assert `v' > 0
}

foreach v of local OUTPUTS {
    capture confirm variable `v'
    if _rc {
        display as error "Required output variable `v' was not found."
        exit 111
    }
    capture confirm numeric variable `v'
    if _rc {
        destring `v', replace ignore(",")
    }
    assert !missing(`v')
    assert `v' > 0
}

* Unique and balanced county-year panel.
isid location_name year, sort
bysort location_name: assert _N == 2
bysort location_name: egen byte __has2014 = max(year == 2014)
bysort location_name: egen byte __has2022 = max(year == 2022)
assert __has2014 == 1 & __has2022 == 1
drop __has2014 __has2022

encode location_name, gen(county_id)
isid county_id year, sort
xtset county_id year

quietly levelsof county_id, local(COUNTY_LEVELS)
local N_COUNTIES : word count `COUNTY_LEVELS'
display as result "Balanced panel: `N_COUNTIES' counties and 2 periods."

* Do not log-transform, min-max normalize, or winsorize the DEA variables.
* Multiplying or dividing each variable by a positive unit constant is harmless;
* nonlinear transformations and translations change the DEA technology.

tempfile ANALYSIS_PANEL
save `ANALYSIS_PANEL', replace

********************************************************************************
* 4. POINT ESTIMATES
********************************************************************************

local RAW_BASE "${OUT}/malmquist_`MODEL'_raw"

quietly malmq2 `INPUTS' = `OUTPUTS', ///
    ort(out) fgnz dmu(location_name) ///
    saving("`RAW_BASE'", replace)

use "`RAW_BASE'.dta", clear

foreach v in TFPCH TECH TECCH SECH {
    capture confirm variable `v'
    if _rc {
        display as error "malmq2 did not return `v'."
        exit 498
    }
}

count if missing(TFPCH, TECH, TECCH, SECH)
if r(N) > 0 {
    display as error "malmq2 returned missing point estimates for " r(N) " counties."
    list location_name county_id Pdwise TFPCH TECH TECCH SECH if missing(TFPCH, TECH, TECCH, SECH), noobs
    exit 498
}

* Under fgnz, malmq2 returns:
*   TFPCH = CRS total factor productivity change
*   TECH  = VRS pure technical efficiency change
*   TECCH = CRS technological change
*   SECH  = scale efficiency change
* Therefore CRS efficiency change equals TECH * SECH.
rename TFPCH Malmquist_Index
rename TECH Pure_Technical_Efficiency_Change
rename TECCH Technical_Change
rename SECH Scale_Efficiency_Change

gen double Efficiency_Change = ///
    Pure_Technical_Efficiency_Change * Scale_Efficiency_Change

gen double decomposition_error = abs( ///
    Malmquist_Index - Efficiency_Change * Technical_Change)

quietly summarize decomposition_error, meanonly
if r(max) > 1e-6 {
    display as error "Malmquist decomposition check failed. Maximum error = " r(max)
    exit 498
}

gen int From_Year = 2014
gen int To_Year   = 2022
gen byte Years    = To_Year - From_Year
gen double Annualized_MPI = Malmquist_Index^(1 / Years)
gen double Annualized_Productivity_Pct = 100 * (Annualized_MPI - 1)

label variable Malmquist_Index "Malmquist productivity change; >1 growth"
label variable Efficiency_Change "CRS efficiency change; TECH x SECH"
label variable Technical_Change "CRS technological change"
label variable Pure_Technical_Efficiency_Change "VRS pure technical efficiency change"
label variable Scale_Efficiency_Change "Scale efficiency change"
label variable Annualized_Productivity_Pct "Annualized productivity change (%)"

order location_name county_id From_Year To_Year Years ///
    Malmquist_Index Efficiency_Change Technical_Change ///
    Pure_Technical_Efficiency_Change Scale_Efficiency_Change ///
    Annualized_MPI Annualized_Productivity_Pct Pdwise
sort location_name

format Malmquist_Index Efficiency_Change Technical_Change ///
    Pure_Technical_Efficiency_Change Scale_Efficiency_Change ///
    Annualized_MPI %12.8f
format Annualized_Productivity_Pct %12.4f

save "${OUT}/malmquist_`MODEL'_point_estimates.dta", replace
export delimited using ///
    "${OUT}/malmquist_`MODEL'_point_estimates.csv", replace

tempfile POINT_ESTIMATES
save `POINT_ESTIMATES', replace

********************************************************************************
* 5. AGGREGATE POINT-ESTIMATE SUMMARY
********************************************************************************

use `POINT_ESTIMATES', clear

tempname SUMMARY_POST
postfile `SUMMARY_POST' str40 Metric ///
    double Arithmetic_Mean Geometric_Mean Annualized_Index ///
    Annualized_Change_Pct using ///
    "${OUT}/malmquist_`MODEL'_summary.dta", replace

foreach v in Malmquist_Index Efficiency_Change Technical_Change ///
             Pure_Technical_Efficiency_Change Scale_Efficiency_Change {
    quietly summarize `v', meanonly
    local AM = r(mean)

    tempvar __ln
    generate double `__ln' = ln(`v')
    quietly summarize `__ln', meanonly
    local GM = exp(r(mean))
    local AI = `GM'^(1/8)
    local AP = 100 * (`AI' - 1)

    post `SUMMARY_POST' ("`v'") (`AM') (`GM') (`AI') (`AP')
    drop `__ln'
}
postclose `SUMMARY_POST'

preserve
use "${OUT}/malmquist_`MODEL'_summary.dta", clear
format Arithmetic_Mean Geometric_Mean Annualized_Index %12.8f
format Annualized_Change_Pct %12.4f
list, noobs abbreviate(32)
export delimited using ///
    "${OUT}/malmquist_`MODEL'_summary.csv", replace
restore

********************************************************************************
* 6. OPTIONAL PAIRED FRONTIER-RESAMPLING BOOTSTRAP
********************************************************************************

if `RUN_BOOTSTRAP' == 1 {

    display as text _newline ///
        "Starting paired county frontier-resampling bootstrap: `BOOT_REPS' replications."
    display as text ///
        "This is a sensitivity bootstrap, not the Simar-Wilson smooth bootstrap."
    display as text ///
        "Intervals are generated for MPI, CRS efficiency change, and technological change."
    display as text ///
        "FGNZ pure-efficiency and scale-change point estimates are retained without bootstrap CIs."

    * malmq2 has already loaded its shepdf program, which is used below to
    * evaluate fixed original counties against each resampled reference frontier.
    capture quietly program list shepdf
    if _rc {
        display as error "The internal shepdf program was not loaded by malmq2."
        exit 499
    }

    tempfile BOOT_REFERENCE
    tempname BOOT_POST

    postfile `BOOT_POST' long replication int county_id ///
        double mpi eff_change tech_change ///
        using "${OUT}/malmquist_`MODEL'_bootstrap_replicates.dta", replace

    set seed `BOOT_SEED'

    forvalues b = 1/`BOOT_REPS' {

        * Resample complete county histories. Both years for a county travel
        * together, preserving the temporal pairing.
        quietly use `ANALYSIS_PANEL', clear
        quietly bsample `N_COUNTIES', ///
            cluster(county_id) idcluster(bootstrap_county_id)
        quietly generate byte __reference = 1
        quietly count
        assert r(N) == 2 * `N_COUNTIES'
        quietly bysort bootstrap_county_id: assert _N == 2
        quietly bysort bootstrap_county_id: egen byte __has2014b = max(year == 2014)
        quietly bysort bootstrap_county_id: egen byte __has2022b = max(year == 2022)
        assert __has2014b == 1 & __has2022b == 1
        drop __has2014b __has2022b
        quietly save `BOOT_REFERENCE', replace

        * Append the resampled reference sample to the unchanged original panel.
        * Original observations are evaluation points; resampled observations
        * define the reference frontiers.
        quietly use `ANALYSIS_PANEL', clear
        quietly generate byte __reference = 0
        quietly append using `BOOT_REFERENCE'

        quietly generate byte __ref2014 = (__reference == 1 & year == 2014)
        quietly generate byte __ref2022 = (__reference == 1 & year == 2022)

        * CRS distances required for the conventional Malmquist index.
        quietly shepdf if __reference == 0 & year == 2014, ///
            gen(__d00c) invars(`INPUTS') opvars(`OUTPUTS') ///
            rflag(__ref2014) ort(out) maxiter(16000) tol(1e-8)

        quietly shepdf if __reference == 0 & year == 2022, ///
            gen(__d10c) invars(`INPUTS') opvars(`OUTPUTS') ///
            rflag(__ref2014) ort(out) maxiter(16000) tol(1e-8)

        quietly shepdf if __reference == 0 & year == 2022, ///
            gen(__d11c) invars(`INPUTS') opvars(`OUTPUTS') ///
            rflag(__ref2022) ort(out) maxiter(16000) tol(1e-8)

        quietly shepdf if __reference == 0 & year == 2014, ///
            gen(__d01c) invars(`INPUTS') opvars(`OUTPUTS') ///
            rflag(__ref2022) ort(out) maxiter(16000) tol(1e-8)

        quietly keep if __reference == 0
        quietly count
        assert r(N) == 2 * `N_COUNTIES'

        * Put the four CRS distance functions on one county row.
        quietly bysort county_id: egen double __D00C = max(__d00c)
        quietly bysort county_id: egen double __D10C = max(__d10c)
        quietly bysort county_id: egen double __D11C = max(__d11c)
        quietly bysort county_id: egen double __D01C = max(__d01c)
        quietly keep if year == 2022

        quietly generate double eff_change = __D11C / __D00C
        quietly generate double tech_change = sqrt( ///
            (__D10C / __D11C) * (__D00C / __D01C))
        quietly generate double mpi = eff_change * tech_change
        quietly count if missing(mpi, eff_change, tech_change) | ///
            mpi <= 0 | eff_change <= 0 | tech_change <= 0

        if r(N) > 0 {
            display as error ///
                "Bootstrap replication `b' produced invalid distances for " r(N) " counties."
            postclose `BOOT_POST'
            exit 498
        }

        quietly sort county_id
        local NB = _N
        forvalues i = 1/`NB' {
            post `BOOT_POST' (`b') (county_id[`i']) ///
                (mpi[`i']) (eff_change[`i']) (tech_change[`i'])
        }

        if `b' == 1 | mod(`b', 50) == 0 | `b' == `BOOT_REPS' {
            display as text "Completed bootstrap replication `b' of `BOOT_REPS'."
        }
    }

    postclose `BOOT_POST'

    use "${OUT}/malmquist_`MODEL'_bootstrap_replicates.dta", clear
    compress
    save "${OUT}/malmquist_`MODEL'_bootstrap_replicates.dta", replace

    * County-specific percentile intervals.
    local ALPHA = (100 - `CI_LEVEL') / 2
    local UPPER = 100 - `ALPHA'

    quietly levelsof county_id, local(BOOT_COUNTIES)
    tempname CI_POST
    tempfile CI_DATA

    postfile `CI_POST' int county_id int Valid_Replications ///
        double MPI_Boot_Mean MPI_Lower MPI_Upper ///
        Eff_Boot_Mean Eff_Change_Lower Eff_Change_Upper ///
        Tech_Boot_Mean Tech_Change_Lower Tech_Change_Upper ///
        using `CI_DATA', replace

    foreach id of local BOOT_COUNTIES {
        quietly count if county_id == `id' & ///
            !missing(mpi, eff_change, tech_change)
        local NV = r(N)

        quietly summarize mpi if county_id == `id', meanonly
        local MPI_MEAN = r(mean)
        quietly _pctile mpi if county_id == `id', p(`ALPHA' `UPPER')
        local MPI_LO = r(r1)
        local MPI_HI = r(r2)

        quietly summarize eff_change if county_id == `id', meanonly
        local EFF_MEAN = r(mean)
        quietly _pctile eff_change if county_id == `id', p(`ALPHA' `UPPER')
        local EFF_LO = r(r1)
        local EFF_HI = r(r2)

        quietly summarize tech_change if county_id == `id', meanonly
        local TEC_MEAN = r(mean)
        quietly _pctile tech_change if county_id == `id', p(`ALPHA' `UPPER')
        local TEC_LO = r(r1)
        local TEC_HI = r(r2)

        post `CI_POST' (`id') (`NV') ///
            (`MPI_MEAN') (`MPI_LO') (`MPI_HI') ///
            (`EFF_MEAN') (`EFF_LO') (`EFF_HI') ///
            (`TEC_MEAN') (`TEC_LO') (`TEC_HI')
    }
    postclose `CI_POST'

    use `POINT_ESTIMATES', clear
    merge 1:1 county_id using `CI_DATA', assert(match) nogen

    * These flags are descriptive because the intervals are sensitivity
    * intervals, not Simar-Wilson inferential intervals.
    generate byte MPI_CI_Excludes_1 = MPI_Lower > 1 | MPI_Upper < 1
    generate byte Eff_CI_Excludes_1 = ///
        Eff_Change_Lower > 1 | Eff_Change_Upper < 1
    generate byte Tech_CI_Excludes_1 = ///
        Tech_Change_Lower > 1 | Tech_Change_Upper < 1
    label variable MPI_Lower "Bootstrap percentile lower CI: MPI"
    label variable MPI_Upper "Bootstrap percentile upper CI: MPI"
    label variable Eff_Change_Lower "Bootstrap percentile lower CI: efficiency change"
    label variable Eff_Change_Upper "Bootstrap percentile upper CI: efficiency change"
    label variable Tech_Change_Lower "Bootstrap percentile lower CI: technology change"
    label variable Tech_Change_Upper "Bootstrap percentile upper CI: technology change"

    sort location_name
    save "${OUT}/malmquist_`MODEL'_results_with_bootstrap_ci.dta", replace
    export delimited using ///
        "${OUT}/malmquist_`MODEL'_results_with_bootstrap_ci.csv", replace

    * Aggregate geometric-mean bootstrap replicates.
    use "${OUT}/malmquist_`MODEL'_bootstrap_replicates.dta", clear
    generate double ln_mpi   = ln(mpi)
    generate double ln_eff   = ln(eff_change)
    generate double ln_tech  = ln(tech_change)

    collapse (mean) ln_mpi ln_eff ln_tech, by(replication)
    generate double GM_MPI  = exp(ln_mpi)
    generate double GM_Eff  = exp(ln_eff)
    generate double GM_Tech = exp(ln_tech)
    drop ln_*

    save "${OUT}/malmquist_`MODEL'_aggregate_bootstrap_replicates.dta", replace
    export delimited using ///
        "${OUT}/malmquist_`MODEL'_aggregate_bootstrap_replicates.csv", replace

    * Percentile intervals for aggregate geometric means.
    tempname AGG_CI_POST
    postfile `AGG_CI_POST' str32 Metric double Boot_Mean Lower Upper ///
        using "${OUT}/malmquist_`MODEL'_aggregate_bootstrap_ci.dta", replace

    quietly summarize GM_MPI, meanonly
    local AGM = r(mean)
    quietly _pctile GM_MPI, p(`ALPHA' `UPPER')
    post `AGG_CI_POST' ("Malmquist_Index") (`AGM') (r(r1)) (r(r2))

    quietly summarize GM_Eff, meanonly
    local AGE = r(mean)
    quietly _pctile GM_Eff, p(`ALPHA' `UPPER')
    post `AGG_CI_POST' ("Efficiency_Change") (`AGE') (r(r1)) (r(r2))

    quietly summarize GM_Tech, meanonly
    local AGT = r(mean)
    quietly _pctile GM_Tech, p(`ALPHA' `UPPER')
    post `AGG_CI_POST' ("Technical_Change") (`AGT') (r(r1)) (r(r2))

    postclose `AGG_CI_POST'

    use "${OUT}/malmquist_`MODEL'_aggregate_bootstrap_ci.dta", clear
    format Boot_Mean Lower Upper %12.8f
    list, noobs
    export delimited using ///
        "${OUT}/malmquist_`MODEL'_aggregate_bootstrap_ci.csv", replace

    display as result _newline ///
        "Bootstrap results saved in ${OUT}."
}

********************************************************************************
* 7. CREATE AN ANALYSIS-READY COUNTY FILE FOR MAPS AND SECOND-STAGE MODELS
********************************************************************************

* Attach the productivity results to the 2014 county covariates. This replaces
* the force merges in the original do-file with an exact, checked 1:1 merge.
tempfile FINAL_RESULTS

if `RUN_BOOTSTRAP' == 1 {
    use "${OUT}/malmquist_`MODEL'_results_with_bootstrap_ci.dta", clear
}
else {
    use `POINT_ESTIMATES', clear
}
save `FINAL_RESULTS', replace

use `ANALYSIS_PANEL', clear
keep if year == 2014
isid county_id
merge 1:1 county_id using `FINAL_RESULTS', assert(match) nogen
sort location_name

save "${OUT}/malmquist_`MODEL'_analysis_ready.dta", replace
export delimited using ///
    "${OUT}/malmquist_`MODEL'_analysis_ready.csv", replace

********************************************************************************
* 8. VALIDATION TARGETS FOR THE SUPPLIED MYDATA.CSV COMBINED MODEL
********************************************************************************

* With INPUTS = pop_dev hw_updated and
* OUTPUTS = childsurvival mumsurvival immunization, the point estimates should
* be approximately:
*
*   Geometric mean Malmquist index       0.382939
*   Geometric mean CRS efficiency change 1.886376
*   Geometric mean technological change  0.203002
*
* The identity Malmquist = efficiency change x technological change must hold.
********************************************************************************

exit
