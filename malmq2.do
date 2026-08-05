/*****************************************************************************
KENYA COUNTY MALMQUIST PRODUCTIVITY ANALYSIS
COUNTY-SPECIFIC 95% CONFIDENCE INTERVALS

PRIMARY SPECIFICATION
  - Original positive input and output levels.
  - No logarithmic transformations.
  - No winsorization or trimming.
  - No capping or rounding before estimation or classification.

COUNTY-LEVEL RESULTS
  For every county and outcome model, this file reports:
    TFPCH  = total factor productivity change
    TECH   = pure technical-efficiency change under FGNZ
    TECCH  = technological/frontier change under FGNZ
    SECH   = scale-efficiency change under FGNZ

  It also reports a county-specific bootstrap confidence interval for every
  indicator. At the default level of 95%:
    lower bound > 1  -> statistically significant improvement
    upper bound < 1  -> statistically significant deterioration
    interval includes 1 -> no statistically distinguishable change

WHY AN ORDINARY CLUSTER BOOTSTRAP IS NOT USED
  A bootstrap that samples counties with replacement omits some named counties
  and duplicates others in each replication. That is suitable for uncertainty
  around an aggregate mean, but it does not yield a complete distribution for
  every original county.

BOOTSTRAP USED HERE
  This file implements the county-specific smoothed Malmquist bootstrap logic:

  1. Estimate paired contemporaneous CRS output-distance scores for 2014 and
     2022.
  2. Resample and smooth those paired scores jointly, preserving their
     intertemporal association.
  3. Construct a complete pseudo-reference panel containing all 47 counties.
  4. Evaluate every county's ORIGINAL 2014 and 2022 production plans against
     the pseudo frontiers.
  5. Recalculate FGNZ TFPCH, TECH, TECCH, and SECH.
  6. Repeat and form a county-specific distribution for every indicator.

  The important distinction in step 4 is that pseudo-observations define each
  bootstrap frontier, while the original county observations are evaluated
  against that frontier. Scoring pseudo-observations against themselves would
  not reproduce the required bootstrap estimator.

PRIMARY CONFIDENCE INTERVAL
  For indicator theta, the basic/reverse-percentile interval is:

      error_b = theta_boot_b - theta_original
      lower   = theta_original - q97.5(error_b)
      upper   = theta_original - q2.5(error_b)

  Percentile intervals are retained as sensitivity results.

IMPORTANT METHOD NOTE
  malmq2 does not provide county-level confidence intervals internally. This
  do-file uses malmq2 for the original point estimates and its underlying
  Shephard-distance routine for bootstrap frontier evaluation. The original
  direct-distance calculations are checked against malmq2 before bootstrapping.

REQUIRED USER-WRITTEN COMMAND
    ssc install malmq2

STATA REQUIREMENT
    Stata 16 or newer.
*****************************************************************************/

version 16.0
clear all
set more off
set type double


/*****************************************************************************
1. USER SETTINGS
*****************************************************************************/

* Replace with a full path when the CSV is not in the working directory.
local datafile "mydata(1).csv"

* Folder for all outputs. Create it before running when it does not exist.
local outdir "."

* Bootstrap replications.
* Use 49 or 99 only to test syntax and paths.
* Use 1,999 or more for final reported confidence intervals.
local B 1999

* Reproducible starting seed. Each outcome model gets a different offset.
local seed 20260804

* Confidence level.
local level 95

* Kernel-bandwidth multiplier. Main analysis: 1.
local hfactor 1

* Maximum attempted draws per model. Zero sets an automatic maximum.
local maxattempts 0

* Models to run. For a first test, use: local models "child"
local models "child maternal hale combined"

* Workforce input. The final CSV also contains hw_updated.
local workforce "healthworkforce"

* Inputs in original positive levels.
local inputs "pop_dev `workforce'"

* Outputs in original positive levels.
local child_output    "childsurvival"
local maternal_output "mumsurvival"
local hale_output     "hale"

* Regional reporting choice only; it does not affect the county bootstrap.
* 1 combines Nairobi with Central, matching the submitted regional summaries.
local combine_nairobi 1


/*****************************************************************************
2. LOAD AND VALIDATE THE FINAL COUNTY PANEL
*****************************************************************************/

capture confirm file "`datafile'"
if _rc {
    display as error "Data file not found: `datafile'"
    exit 601
}

import delimited using "`datafile'", clear varnames(1) case(preserve)

* county is a string in the attached final CSV.
capture drop county_id region_id Region_analysis
egen long county_id = group(county), label

generate str20 Region_analysis = Region

if `combine_nairobi' == 1 {
    replace Region_analysis = "Central" if Region_analysis == "Nairobi"
}
else if `combine_nairobi' != 0 {
    display as error "combine_nairobi must be 0 or 1"
    exit 198
}

egen int region_id = group(Region_analysis), label

* Required balanced 47-county, two-period panel.
isid county_id year
assert inlist(year, 2014, 2022)
bysort county_id (year): assert _N == 2
bysort county_id (year): assert year[1] == 2014 & year[2] == 2022
bysort county_id (year): assert region_id == region_id[1]

quietly count
assert r(N) == 94

quietly levelsof county_id, local(county_levels)
local ncounty : word count `county_levels'
assert `ncounty' == 47

* Confirm DEA variables are numeric, nonmissing, and strictly positive.
foreach v in `inputs' `child_output' `maternal_output' `hale_output' {

    capture confirm numeric variable `v'
    if _rc {
        display as error "Required numeric variable not found: `v'"
        exit 111
    }

    quietly count if missing(`v') | `v' <= 0
    if r(N) > 0 {
        display as error "`v' has missing/nonpositive observations: " r(N)
        exit 459
    }
}

* These are inverse-mortality indices inherited from the submitted analysis.
* They are not literal survival rates.
quietly count if abs(childsurvival - 1000/under5) > 1e-5
assert r(N) == 0

quietly count if abs(mumsurvival - 1000/mmr) > 1e-5
assert r(N) == 0

capture which malmq2
if _rc {
    display as error "malmq2 is not installed. Run: ssc install malmq2"
    exit 199
}

xtset county_id year

* Clean panel used by every model and replication.
tempfile analysis_panel
save `analysis_panel', replace

* County metadata for final tables.
preserve
    keep if year == 2014
    keep county_id county Region Region_analysis region_id
    isid county_id
    sort county_id

    tempfile county_metadata
    save `county_metadata', replace

    export delimited using "`outdir'/county_crosswalk.csv", replace
restore


/*****************************************************************************
3. EVALUATE ORIGINAL TARGETS AGAINST A SPECIFIED REFERENCE PANEL

This program reproduces the FGNZ calculations used by malmq2, while allowing
separate target observations and reference observations. That separation is
required for the county-specific frontier bootstrap.
*****************************************************************************/

capture program drop fgnz_against_reference
program define fgnz_against_reference

    version 16.0

    syntax, ///
        TARGETS(string) ///
        REFERENCES(string) ///
        INPUTS(string asis) ///
        OUTPUTS(string asis) ///
        SAVING(string)

    tempfile target_data

    * Original observations to be evaluated.
    use "`targets'", clear
    keep county_id year `inputs' `outputs'
    isid county_id year

    generate byte __target = 1
    generate byte __reference = 0
    save `target_data', replace

    * Observations defining the production frontiers.
    use "`references'", clear
    keep county_id year `inputs' `outputs'
    isid county_id year

    generate byte __target = 0
    generate byte __reference = 1

    append using `target_data'

    generate byte __rflag = 0
    generate double Dsame  = .
    generate double Dcross = .
    generate double Dvrs   = .

    foreach yy in 2014 2022 {

        if `yy' == 2014 {
            local other 2022
        }
        else {
            local other 2014
        }

        * CRS contemporaneous distance: target yy relative to frontier yy.
        replace __rflag = (__reference == 1 & year == `yy')

        tempvar dsame
        quietly shepdf if __target == 1 & year == `yy', ///
            gen(`dsame') ///
            invars(`inputs') ///
            opvars(`outputs') ///
            rflag(__rflag) ///
            ort(o) ///
            maxiter(16000) ///
            tol(1e-8)

        replace Dsame = `dsame' if __target == 1 & year == `yy'
        drop `dsame'

        * CRS cross-period distance: target yy relative to the other frontier.
        replace __rflag = (__reference == 1 & year == `other')

        tempvar dcross
        quietly shepdf if __target == 1 & year == `yy', ///
            gen(`dcross') ///
            invars(`inputs') ///
            opvars(`outputs') ///
            rflag(__rflag) ///
            ort(o) ///
            maxiter(16000) ///
            tol(1e-8)

        replace Dcross = `dcross' if __target == 1 & year == `yy'
        drop `dcross'

        * VRS contemporaneous distance for pure efficiency change.
        replace __rflag = (__reference == 1 & year == `yy')

        tempvar dvrs
        quietly shepdf if __target == 1 & year == `yy', ///
            gen(`dvrs') ///
            invars(`inputs') ///
            opvars(`outputs') ///
            rflag(__rflag) ///
            ort(o) ///
            vrs ///
            maxiter(16000) ///
            tol(1e-8)

        replace Dvrs = `dvrs' if __target == 1 & year == `yy'
        drop `dvrs'
    }

    keep if __target == 1
    keep county_id year Dsame Dcross Dvrs

    isid county_id year

    quietly count if ///
        missing(Dsame, Dcross, Dvrs) | ///
        Dsame <= 0 | Dcross <= 0 | Dvrs <= 0

    if r(N) > 0 {
        display as error ///
            "Infeasible or nonpositive distance functions: " r(N)
        exit 498
    }

    reshape wide Dsame Dcross Dvrs, i(county_id) j(year)

    * FGNZ decomposition, matching malmq2's output-oriented implementation.
    generate double TECH_crs = Dsame2022/Dsame2014

    generate double TECH = Dvrs2022/Dvrs2014

    generate double TECCH = sqrt( ///
        (Dcross2022/Dsame2022) * ///
        (Dsame2014/Dcross2014) ///
    )

    generate double TFPCH = TECH_crs*TECCH
    generate double SECH  = TECH_crs/TECH

    quietly count if ///
        missing(TFPCH, TECH, TECCH, SECH) | ///
        TFPCH <= 0 | TECH <= 0 | TECCH <= 0 | SECH <= 0

    if r(N) > 0 {
        display as error "Invalid FGNZ indices: " r(N)
        exit 498
    }

    quietly count if ///
        abs(TFPCH - TECH*SECH*TECCH) > ///
        1e-5*max(1, abs(TFPCH))

    if r(N) > 0 {
        display as error "FGNZ decomposition identity failed."
        exit 498
    }

    rename Dsame2014 DCRS2014
    rename Dsame2022 DCRS2022

    keep ///
        county_id ///
        TFPCH TECH TECCH SECH ///
        DCRS2014 DCRS2022

    sort county_id
    save "`saving'", replace

end


/*****************************************************************************
4. MODEL-SPECIFIC COUNTY BOOTSTRAP PROGRAM
*****************************************************************************/

capture program drop malmq_county_ci
program define malmq_county_ci

    version 16.0

    syntax, ///
        MODEL(string) ///
        PANEL(string) ///
        INPUTS(string asis) ///
        OUTPUTS(string asis) ///
        OUTDIR(string) ///
        REPS(integer) ///
        SEED(integer) ///
        METADATA(string) ///
        [LEVEL(real 95) HFACTOR(real 1) MAXATTEMPTS(integer 0)]

    if `reps' < 19 {
        display as error "Use at least 19 bootstrap replications."
        exit 198
    }

    if `level' <= 0 | `level' >= 100 {
        display as error "level() must be between 0 and 100."
        exit 198
    }

    if `hfactor' <= 0 {
        display as error "hfactor() must be positive."
        exit 198
    }

    if `maxattempts' == 0 {
        local maxattempts = ceil(3*`reps') + 200
    }

    local plo = (100 - `level')/2
    local phi = 100 - `plo'

    display as text _newline(2) ///
        "============================================================"
    display as text "County bootstrap model: `model'"
    display as text "Inputs:  `inputs'"
    display as text "Outputs: `outputs'"
    display as text "Requested complete replications: `reps'"
    display as text ///
        "============================================================"


    /*************************************************************************
    4A. ORIGINAL MALMQ2 POINT ESTIMATES
    *************************************************************************/

    use "`panel'", clear

    quietly levelsof county_id, local(expected_counties)
    local expected_n : word count `expected_counties'

    xtset county_id year

    tempfile point_raw point_for_merge point_direct

    quietly malmq2 `inputs' = `outputs', ///
        ort(o) ///
        fgnz ///
        dmu(county) ///
        saving("`point_raw'", replace)

    use "`point_raw'", clear
    isid county_id

    quietly count
    local ncounty = r(N)

    if `ncounty' != `expected_n' {
        display as error ///
            "Original malmq2 result does not contain every county. " ///
            "Expected `expected_n'; found `ncounty'."
        exit 498
    }

    quietly count if ///
        missing(TFPCH, TECH, TECCH, SECH) | ///
        TFPCH <= 0 | TECH <= 0 | TECCH <= 0 | SECH <= 0

    if r(N) > 0 {
        display as error ///
            "Original malmq2 estimates contain missing/nonpositive values."
        exit 498
    }

    quietly count if ///
        abs(TFPCH - TECH*SECH*TECCH) > ///
        1e-5*max(1, abs(TFPCH))

    if r(N) > 0 {
        display as error "Original FGNZ identity failed."
        exit 498
    }

    * Save point estimates with county labels. The malmq2 result already
    * contains the dmu() variable, so drop it before merging the validated
    * metadata copy to avoid overlapping-variable ambiguity.
    keep county_id TFPCH TECH TECCH SECH
    merge 1:1 county_id using "`metadata'", assert(match) nogen

    generate str12 model = "`model'"
    order model county_id county Region Region_analysis region_id ///
        TFPCH TECH TECCH SECH

    format TFPCH TECH TECCH SECH %12.6f

    save "`outdir'/productivity_`model'_levels.dta", replace
    export delimited using ///
        "`outdir'/productivity_`model'_levels.csv", replace

    keep county_id TFPCH TECH TECCH SECH
    rename TFPCH point_TFPCH
    rename TECH  point_TECH
    rename TECCH point_TECCH
    rename SECH  point_SECH
    save `point_for_merge', replace


    /*************************************************************************
    4B. VERIFY DIRECT DISTANCES AGAINST MALMQ2

    This also supplies the paired CRS contemporaneous distances used to build
    the smoothed pseudo-reference samples.
    *************************************************************************/

    quietly fgnz_against_reference, ///
        targets("`panel'") ///
        references("`panel'") ///
        inputs(`inputs') ///
        outputs(`outputs') ///
        saving("`point_direct'")

    use "`point_direct'", clear

    rename TFPCH direct_TFPCH
    rename TECH  direct_TECH
    rename TECCH direct_TECCH
    rename SECH  direct_SECH

    merge 1:1 county_id using `point_for_merge', assert(match) nogen

    foreach index in TFPCH TECH TECCH SECH {

        quietly count if ///
            abs(direct_`index' - point_`index') > ///
            1e-5*max(1, abs(point_`index'))

        if r(N) > 0 {
            display as error ///
                "Direct distance calculation does not reproduce malmq2: " ///
                "`index'"
            exit 498
        }
    }

    display as result ///
        "Direct FGNZ calculations reproduce malmq2 point estimates."


    /*************************************************************************
    4C. PREPARE PAIRED CRS DISTANCES AND THE ORIGINAL MODEL PANEL
    *************************************************************************/

    tempfile distance_pairs distance_long model_panel county_targets

    preserve
        keep county_id DCRS2014 DCRS2022
        rename DCRS2014 D2014
        rename DCRS2022 D2022
        isid county_id
        sort county_id
        generate long target_order = _n
        save `distance_pairs', replace

        keep target_order county_id
        save `county_targets', replace
    restore

    preserve
        keep county_id DCRS2014 DCRS2022
        rename DCRS2014 d_crs2014
        rename DCRS2022 d_crs2022
        reshape long d_crs, i(county_id) j(year)
        isid county_id year
        save `distance_long', replace
    restore

    use "`panel'", clear
    merge 1:1 county_id year using `distance_long', assert(match) nogen

    quietly count if missing(d_crs) | d_crs <= 0 | d_crs > 1 + 1e-6
    if r(N) > 0 {
        display as error "Invalid original CRS distance scores: " r(N)
        exit 498
    }

    replace d_crs = 1 if d_crs > 1 & d_crs <= 1 + 1e-6
    save `model_panel', replace


    /*************************************************************************
    4D. BIVARIATE SMOOTHING PARAMETERS

    Gaussian smoothing is applied to the paired 2014/2022 CRS distances.
    With two dimensions, the scalar bandwidth is n^(-1/6). The covariance
    matrix preserves the scale and association of the paired distances.
    Reflection enforces the support (0,1].
    *************************************************************************/

    use `distance_pairs', clear

    quietly summarize D2014, meanonly
    tempname mu14 mu22 c11 c12 c22 ridge l11 l21 l22 h denom
    scalar `mu14' = r(mean)

    quietly summarize D2022, meanonly
    scalar `mu22' = r(mean)

    quietly correlate D2014 D2022, covariance
    tempname C
    matrix `C' = r(C)

    scalar `c11' = `C'[1,1]
    scalar `c12' = `C'[1,2]
    scalar `c22' = `C'[2,2]

    * Tiny ridge for a stable Cholesky factor in degenerate cases.
    scalar `ridge' = max(1e-12, 1e-10*max(`c11', `c22'))
    scalar `l11' = sqrt(`c11' + `ridge')
    scalar `l21' = `c12'/`l11'
    scalar `l22' = sqrt(max(`c22' + `ridge' - `l21'^2, `ridge'))

    scalar `h' = `hfactor' * (`ncounty'^(-1/6))
    scalar `denom' = sqrt(1 + `h'^2)

    display as result ///
        "Bivariate smoothing bandwidth = " %9.6f `h'


    /*************************************************************************
    4E. COUNTY-PRESERVING SMOOTHED FRONTIER BOOTSTRAP
    *************************************************************************/

    tempfile stars pseudo_reference boot_result boot_draws
    tempname POSTH

    postfile `POSTH' ///
        long rep ///
        long county_id ///
        double TFPCH TECH TECCH SECH ///
        using `boot_draws', replace

    set seed `seed'

    local accepted 0
    local attempted 0

    while `accepted' < `reps' & `attempted' < `maxattempts' {

        local ++attempted

        * Draw one paired donor vector for each target county. Target county
        * identifiers remain fixed and complete in every replication.
        use `distance_pairs', clear
        bsample `ncounty'

        keep D2014 D2022
        rename D2014 donor2014
        rename D2022 donor2022
        generate long target_order = _n

        merge 1:1 target_order using `county_targets', ///
            assert(match) nogen

        generate double z1 = rnormal()
        generate double z2 = rnormal()

        generate double smooth2014 = ///
            donor2014 + `h'*(`l11'*z1)

        generate double smooth2022 = ///
            donor2022 + `h'*(`l21'*z1 + `l22'*z2)

        * Boundary reflection for support [0,1].
        forvalues rr = 1/20 {
            replace smooth2014 = -smooth2014 if smooth2014 < 0
            replace smooth2014 = 2 - smooth2014 if smooth2014 > 1
            replace smooth2022 = -smooth2022 if smooth2022 < 0
            replace smooth2022 = 2 - smooth2022 if smooth2022 > 1
        }

        * Variance correction after kernel smoothing.
        generate double dstar2014 = ///
            `mu14' + (smooth2014 - `mu14')/`denom'

        generate double dstar2022 = ///
            `mu22' + (smooth2022 - `mu22')/`denom'

        * Reflect again after variance correction.
        forvalues rr = 1/20 {
            replace dstar2014 = -dstar2014 if dstar2014 < 0
            replace dstar2014 = 2 - dstar2014 if dstar2014 > 1
            replace dstar2022 = -dstar2022 if dstar2022 < 0
            replace dstar2022 = 2 - dstar2022 if dstar2022 > 1
        }

        replace dstar2014 = 1e-8 if dstar2014 <= 0
        replace dstar2014 = 1     if dstar2014 > 1
        replace dstar2022 = 1e-8 if dstar2022 <= 0
        replace dstar2022 = 1     if dstar2022 > 1

        keep county_id dstar2014 dstar2022
        reshape long dstar, i(county_id) j(year)
        isid county_id year
        save `stars', replace

        * Build the complete pseudo-reference panel. The original point is
        * projected to the CRS frontier by y/d_crs and then moved inward by
        * the smoothed pseudo-distance dstar.
        use `model_panel', clear
        merge 1:1 county_id year using `stars', assert(match) nogen

        foreach y of local outputs {
            replace `y' = `y' * dstar/d_crs
        }

        quietly count if missing(dstar) | dstar <= 0
        local good = (r(N) == 0)

        foreach v of local inputs {
            if `good' {
                quietly count if missing(`v') | `v' <= 0
                if r(N) > 0 {
                    local good 0
                }
            }
        }

        foreach v of local outputs {
            if `good' {
                quietly count if missing(`v') | `v' <= 0
                if r(N) > 0 {
                    local good 0
                }
            }
        }

        if `good' {
            keep county_id county year `inputs' `outputs'
            isid county_id year
            save `pseudo_reference', replace

            capture quietly fgnz_against_reference, ///
                targets("`panel'") ///
                references("`pseudo_reference'") ///
                inputs(`inputs') ///
                outputs(`outputs') ///
                saving("`boot_result'")

            if _rc {
                local good 0
            }
        }

        if `good' {
            capture quietly use "`boot_result'", clear
            if _rc {
                local good 0
            }
        }

        if `good' {
            capture quietly isid county_id
            if _rc {
                local good 0
            }
        }

        if `good' {
            quietly count
            if r(N) != `ncounty' {
                local good 0
            }
        }

        if `good' {
            quietly count if ///
                missing(TFPCH, TECH, TECCH, SECH) | ///
                TFPCH <= 0 | TECH <= 0 | TECCH <= 0 | SECH <= 0

            if r(N) > 0 {
                local good 0
            }
        }

        if `good' {
            quietly count if ///
                abs(TFPCH - TECH*SECH*TECCH) > ///
                1e-5*max(1, abs(TFPCH))

            if r(N) > 0 {
                local good 0
            }
        }

        if `good' {

            local ++accepted
            sort county_id

            forvalues ii = 1/`ncounty' {

                post `POSTH' ///
                    (`accepted') ///
                    (county_id[`ii']) ///
                    (TFPCH[`ii']) ///
                    (TECH[`ii']) ///
                    (TECCH[`ii']) ///
                    (SECH[`ii'])
            }

            if mod(`accepted', 50) == 0 | `accepted' == `reps' {
                display as text ///
                    "Accepted replications: `accepted' of `reps'" ///
                    "  (attempts: `attempted')"
            }
        }
    }

    postclose `POSTH'

    if `accepted' < `reps' {
        display as error ///
            "Only `accepted' complete replications were accepted after " ///
            "`attempted' attempts. Increase maxattempts() or inspect " ///
            "infeasible cross-period distance functions."
        exit 498
    }

    use `boot_draws', clear
    isid rep county_id
    sort rep county_id

    save ///
        "`outdir'/county_boot_draws_`model'_levels.dta", replace

    export delimited using ///
        "`outdir'/county_boot_draws_`model'_levels.csv", replace


    /*************************************************************************
    4F. COUNTY-SPECIFIC CONFIDENCE INTERVALS
    *************************************************************************/

    merge m:1 county_id using `point_for_merge', assert(match) nogen

    foreach index in TFPCH TECH TECCH SECH {

        generate double error_`index' = ///
            `index' - point_`index'

        bysort county_id: egen double `index'_bootmean = ///
            mean(`index')

        bysort county_id: egen double `index'_bse = ///
            sd(`index')

        bysort county_id: egen double __elo_`index' = ///
            pctile(error_`index'), p(`plo')

        bysort county_id: egen double __ehi_`index' = ///
            pctile(error_`index'), p(`phi')

        * Primary basic/reverse-percentile interval.
        generate double `index'_lb = ///
            point_`index' - __ehi_`index'

        generate double `index'_ub = ///
            point_`index' - __elo_`index'

        * Percentile interval retained as a sensitivity analysis.
        bysort county_id: egen double `index'_pct_lb = ///
            pctile(`index'), p(`plo')

        bysort county_id: egen double `index'_pct_ub = ///
            pctile(`index'), p(`phi')

        generate double `index'_bias = ///
            `index'_bootmean - point_`index'

        generate double `index'_bc = ///
            point_`index' - `index'_bias

        * Simar-Wilson MSE rule for whether the bias-corrected estimate is
        * preferable to the original point estimate.
        generate byte `index'_bc_preferred = ///
            (`index'_bse^2 < (`index'_bias^2)/3)

        generate byte `index'_significant = ///
            (`index'_lb > 1 | `index'_ub < 1)

        generate str28 `index'_conclusion = ///
            cond(`index'_lb > 1, ///
                 "Significant improvement", ///
            cond(`index'_ub < 1, ///
                 "Significant deterioration", ///
                 "No significant change"))
    }

    bysort county_id: keep if _n == 1

    drop TFPCH TECH TECCH SECH
    drop error_* __elo_* __ehi_*

    rename point_TFPCH TFPCH
    rename point_TECH  TECH
    rename point_TECCH TECCH
    rename point_SECH  SECH

    merge 1:1 county_id using "`metadata'", assert(match) nogen

    generate str12 model = "`model'"
    generate long bootstrap_reps = `accepted'
    generate double ci_level = `level'
    generate str26 ci_method = "basic/reverse-percentile"

    order ///
        model county_id county Region Region_analysis region_id ///
        bootstrap_reps ci_level ci_method ///
        TFPCH TFPCH_lb TFPCH_ub TFPCH_bc TFPCH_bse TFPCH_bias ///
        TFPCH_pct_lb TFPCH_pct_ub ///
        TFPCH_bc_preferred TFPCH_significant TFPCH_conclusion ///
        TECH TECH_lb TECH_ub TECH_bc TECH_bse TECH_bias ///
        TECH_pct_lb TECH_pct_ub ///
        TECH_bc_preferred TECH_significant TECH_conclusion ///
        TECCH TECCH_lb TECCH_ub TECCH_bc TECCH_bse TECCH_bias ///
        TECCH_pct_lb TECCH_pct_ub ///
        TECCH_bc_preferred TECCH_significant TECCH_conclusion ///
        SECH SECH_lb SECH_ub SECH_bc SECH_bse SECH_bias ///
        SECH_pct_lb SECH_pct_ub ///
        SECH_bc_preferred SECH_significant SECH_conclusion

    sort county_id

    format ///
        TFPCH TFPCH_lb TFPCH_ub TFPCH_bc TFPCH_bse TFPCH_bias ///
        TFPCH_pct_lb TFPCH_pct_ub ///
        TECH TECH_lb TECH_ub TECH_bc TECH_bse TECH_bias ///
        TECH_pct_lb TECH_pct_ub ///
        TECCH TECCH_lb TECCH_ub TECCH_bc TECCH_bse TECCH_bias ///
        TECCH_pct_lb TECCH_pct_ub ///
        SECH SECH_lb SECH_ub SECH_bc SECH_bse SECH_bias ///
        SECH_pct_lb SECH_pct_ub ///
        %12.6f

    label variable TFPCH "Original total factor productivity change"
    label variable TFPCH_lb "Basic-bootstrap lower CI bound"
    label variable TFPCH_ub "Basic-bootstrap upper CI bound"

    label variable TECH "Original pure technical-efficiency change"
    label variable TECH_lb "Basic-bootstrap lower CI bound"
    label variable TECH_ub "Basic-bootstrap upper CI bound"

    label variable TECCH "Original technological change"
    label variable TECCH_lb "Basic-bootstrap lower CI bound"
    label variable TECCH_ub "Basic-bootstrap upper CI bound"

    label variable SECH "Original scale-efficiency change"
    label variable SECH_lb "Basic-bootstrap lower CI bound"
    label variable SECH_ub "Basic-bootstrap upper CI bound"

    save "`outdir'/county_ci_`model'_levels.dta", replace

    export delimited using ///
        "`outdir'/county_ci_`model'_levels.csv", replace

    display as result _newline ///
        "County-specific CI file written: " ///
        "`outdir'/county_ci_`model'_levels.csv"

end


/*****************************************************************************
5. RUN THE SELECTED OUTCOME MODELS
*****************************************************************************/

local model_number 0

foreach model of local models {

    local ++model_number

    if "`model'" == "child" {
        local outputs "`child_output'"
    }
    else if "`model'" == "maternal" {
        local outputs "`maternal_output'"
    }
    else if "`model'" == "hale" {
        local outputs "`hale_output'"
    }
    else if "`model'" == "combined" {
        local outputs ///
            "`hale_output' `child_output' `maternal_output'"
    }
    else {
        display as error "Unknown model: `model'"
        exit 198
    }

    local model_seed = `seed' + 10000*`model_number'

    malmq_county_ci, ///
        model(`model') ///
        panel("`analysis_panel'") ///
        inputs(`inputs') ///
        outputs(`outputs') ///
        outdir("`outdir'") ///
        reps(`B') ///
        seed(`model_seed') ///
        metadata("`county_metadata'") ///
        level(`level') ///
        hfactor(`hfactor') ///
        maxattempts(`maxattempts')
}


/*****************************************************************************
6. COMBINE ALL COUNTY CI FILES
*****************************************************************************/

local first_model 1

foreach model of local models {

    if `first_model' == 1 {
        use "`outdir'/county_ci_`model'_levels.dta", clear
        local first_model 0
    }
    else {
        append using "`outdir'/county_ci_`model'_levels.dta"
    }
}

sort model county_id

save "`outdir'/county_ci_all_models_levels.dta", replace
export delimited using ///
    "`outdir'/county_ci_all_models_levels.csv", replace


/*****************************************************************************
7. COMPACT WIDE COUNTY TABLE
*****************************************************************************/

preserve

    keep ///
        model county_id county Region_analysis ///
        TFPCH TFPCH_lb TFPCH_ub TFPCH_conclusion ///
        TECH TECH_lb TECH_ub TECH_conclusion ///
        TECCH TECCH_lb TECCH_ub TECCH_conclusion ///
        SECH SECH_lb SECH_ub SECH_conclusion

    sort model Region_analysis county

    save ///
        "`outdir'/county_ci_compact_all_models_levels.dta", replace

    export delimited using ///
        "`outdir'/county_ci_compact_all_models_levels.csv", replace

restore


/*****************************************************************************
8. LONG COUNTY TABLE: ONE ROW PER MODEL-COUNTY-INDICATOR
*****************************************************************************/

preserve

    keep ///
        model county_id county Region Region_analysis region_id ///
        bootstrap_reps ci_level ci_method ///
        TFPCH TFPCH_lb TFPCH_ub TFPCH_significant TFPCH_conclusion ///
        TECH TECH_lb TECH_ub TECH_significant TECH_conclusion ///
        TECCH TECCH_lb TECCH_ub TECCH_significant TECCH_conclusion ///
        SECH SECH_lb SECH_ub SECH_significant SECH_conclusion

    foreach index in TFPCH TECH TECCH SECH {
        rename `index' point_`index'
        rename `index'_lb lower_`index'
        rename `index'_ub upper_`index'
        rename `index'_significant significant_`index'
        rename `index'_conclusion conclusion_`index'
    }

    reshape long ///
        point_ lower_ upper_ significant_ conclusion_, ///
        i(model county_id) ///
        j(indicator) string

    rename point_ estimate
    rename lower_ ci_lower
    rename upper_ ci_upper
    rename significant_ significant
    rename conclusion_ conclusion

    order ///
        model county_id county Region Region_analysis region_id ///
        indicator estimate ci_lower ci_upper ///
        significant conclusion bootstrap_reps ci_level ci_method

    sort model indicator Region_analysis county

    save ///
        "`outdir'/county_ci_long_all_models_levels.dta", replace

    export delimited using ///
        "`outdir'/county_ci_long_all_models_levels.csv", replace

restore


/*****************************************************************************
9. COUNTS OF SIGNIFICANT COUNTY CHANGES BY MODEL
*****************************************************************************/

preserve

    foreach index in TFPCH TECH TECCH SECH {
        generate byte `index'_improve = (`index'_lb > 1)
        generate byte `index'_decline = (`index'_ub < 1)
        generate byte `index'_nochange = ///
            (`index'_lb <= 1 & `index'_ub >= 1)
    }

    collapse ///
        (sum) ///
            TFPCH_improve TFPCH_decline TFPCH_nochange ///
            TECH_improve TECH_decline TECH_nochange ///
            TECCH_improve TECCH_decline TECCH_nochange ///
            SECH_improve SECH_decline SECH_nochange, ///
        by(model)

    sort model

    save ///
        "`outdir'/county_significance_counts_levels.dta", replace

    export delimited using ///
        "`outdir'/county_significance_counts_levels.csv", replace

restore


display as result _newline(2) ///
    "Completed county-specific Malmquist confidence intervals."

display as result ///
    "Main long result: `outdir'/county_ci_long_all_models_levels.csv"

display as result ///
    "Main wide result: `outdir'/county_ci_all_models_levels.csv"

/*****************************************************************************
END OF FILE
*****************************************************************************/
