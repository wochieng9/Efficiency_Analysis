/*****************************************************************************
KENYA COUNTY EFFICIENCY / PRODUCTIVITY ANALYSIS
PRIMARY SPECIFICATION: ORIGINAL POSITIVE LEVELS ONLY

This version intentionally does NOT:
  - log-transform inputs or outputs;
  - add constants such as +1;
  - winsorize or trim observations; or
  - cap estimated Malmquist components.

It:
  1. validates the 47-county, two-period panel;
  2. estimates output-oriented FGNZ Malmquist indices using malmq2;
  3. saves county point estimates for child, maternal, HALE, and combined
     outcome specifications; and
  4. obtains 95% bootstrap confidence intervals for national and regional
     mean TFPCH, TECH, TECCH, and SECH by resampling intact county panels.

Interpretation of indices:
  TFPCH > 1  productivity improvement
  TFPCH < 1  productivity deterioration
  TECH  > 1  technical-efficiency improvement
  TECCH > 1  favorable technological/frontier change
  SECH  > 1  scale-efficiency improvement

Important output-variable note:
  childsurvival = 1000/under5 and mumsurvival = 1000/mmr are inverse
  mortality indices inherited from the submitted analysis. They are not
  literal survival rates. They are kept here only so that higher values remain
  desirable outputs in malmq2. A model that treats mortality directly as an
  undesirable output requires a different estimator.

Bootstrap note:
  This is a county-panel nonparametric bootstrap for aggregate mean indices.
  It is not the specialized smooth Simar-Wilson DEA-frontier bootstrap and
  should not be described as county-specific frontier bias correction.

Required community command:
  ssc install malmq2

Stata requirement:
  malmq2 requires Stata 16 or newer.
*****************************************************************************/

version 16.0
clear all
set more off
set seed 20260804


/*****************************************************************************
1. USER SETTINGS
*****************************************************************************/

* Replace with a full path when the CSV is not in the working directory.
local datafile "mydata(1).csv"

* Folder in which all results will be saved.
local outdir "."

* Bootstrap settings.
* Use 199 or 499 while checking the code; use 1,999 or more for final results.
local B    1999
local seed 20260804

* Outcome models to estimate.
local models "child maternal hale combined"

* The submitted analysis combined Nairobi with Central for regional reporting.
* Set to 0 to keep Nairobi separate. With one Nairobi county, a separate
* Nairobi-stratum bootstrap interval has no meaningful sampling variation.
local combine_nairobi 1

* Choose the workforce input after confirming its definition and source.
* The final dataset also contains hw_updated.
local workforce "healthworkforce"

* Inputs used in every model, in their original positive levels.
local inputs "pop_dev `workforce'"

* Outputs used in their original positive levels as stored in the final CSV.
local child_output    "childsurvival"
local maternal_output "mumsurvival"
local hale_output     "hale"


/*****************************************************************************
2. LOAD AND VALIDATE THE FINAL PANEL
*****************************************************************************/

capture confirm file "`datafile'"
if _rc {
    display as error "Data file not found: `datafile'"
    exit 601
}

import delimited using "`datafile'", clear varnames(1) case(preserve)

* county is a string in the final CSV; create a separate numeric panel ID.
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

* Confirm unique county-year records and a balanced two-period panel.
isid county_id year
assert inlist(year, 2014, 2022)
bysort county_id: assert _N == 2
bysort county_id (year): assert year[1] == 2014 & year[2] == 2022
bysort county_id (year): assert region_id == region_id[1]

quietly count
assert r(N) == 94

quietly levelsof county_id, local(counties)
local ncounty : word count `counties'
assert `ncounty' == 47

quietly levelsof region_id, local(regions)
local nregion : word count `regions'

xtset county_id year

* DEA requires usable, positive input and output values for this specification.
foreach v in `inputs' `child_output' `maternal_output' `hale_output' {

    capture confirm numeric variable `v'

    if _rc {
        display as error "Required numeric variable not found: `v'"
        exit 111
    }

    quietly count if missing(`v') | `v' <= 0

    if r(N) > 0 {
        display as error "`v' has missing or nonpositive observations: " r(N)
        exit 459
    }
}

* Confirm that the inverse-mortality variables agree with the submitted data
* construction. These checks do not imply that they are literal survival rates.
quietly count if abs(childsurvival - 1000/under5) > 1e-5
assert r(N) == 0

quietly count if abs(mumsurvival - 1000/mmr) > 1e-5
assert r(N) == 0

* Confirm that malmq2 is installed.
capture which malmq2
if _rc {
    display as error "malmq2 is not installed. Run: ssc install malmq2"
    exit 199
}

* Region crosswalk for interpreting r1, r2, etc. in bootstrap output.
preserve
    keep region_id Region_analysis
    duplicates drop
    sort region_id
    export delimited using "`outdir'/region_crosswalk.csv", replace
    list region_id Region_analysis, noobs sep(0)
restore

* Keep a clean copy of the validated full panel in memory for later sections.
tempfile analysis_panel
save `analysis_panel', replace


/*****************************************************************************
3. INTERNAL MALMQUIST ESTIMATION PROGRAM: LEVELS ONLY

The variables supplied in inputs() and outputs() are passed directly to
malmq2. No transformation, winsorization, trimming, or capping is performed.
*****************************************************************************/

capture program drop mpi_compute_levels
program define mpi_compute_levels

    version 16.0

    syntax [if] [in], ///
        INPUTS(varlist numeric min=1) ///
        OUTPUTS(varlist numeric min=1) ///
        ID(varname numeric) ///
        TIME(varname numeric) ///
        DMU(varname) ///
        SAVING(string)

    marksample touse, novarlist

    preserve

        quietly keep if `touse'
        quietly keep `id' `time' `dmu' `inputs' `outputs'

        foreach v of varlist `inputs' `outputs' {

            quietly count if missing(`v') | `v' <= 0

            if r(N) > 0 {
                restore
                display as error "Missing or nonpositive value in `v'"
                exit 459
            }
        }

        capture quietly isid `id' `time'

        if _rc {
            local rc = _rc
            restore
            display as error "Panel identifier is not unique within time."
            exit `rc'
        }

        capture quietly xtset `id' `time'

        if _rc {
            local rc = _rc
            restore
            exit `rc'
        }

        capture quietly malmq2 `inputs' = `outputs', ///
            ort(o) ///
            fgnz ///
            dmu(`dmu') ///
            saving("`saving'", replace)

        if _rc {
            local rc = _rc
            restore
            exit `rc'
        }

    restore

end


/*****************************************************************************
4. COUNTY POINT ESTIMATES
*****************************************************************************/

use `analysis_panel', clear
xtset county_id year

foreach model of local models {

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
        local outputs "`hale_output' `child_output' `maternal_output'"
    }
    else {
        display as error "Unknown model: `model'"
        exit 198
    }

    display as text _newline ///
        "Point estimates using untransformed levels: `model'"

    mpi_compute_levels, ///
        inputs(`inputs') ///
        outputs(`outputs') ///
        id(county_id) ///
        time(year) ///
        dmu(county) ///
        saving("`outdir'/productivity_`model'_levels.dta")
}


/*****************************************************************************
5. BOOTSTRAP STATISTIC PROGRAM

Each bootstrap draw:
  - resamples counties as intact two-year panels within reporting region;
  - assigns a new panel ID when the same county is selected more than once;
  - reruns malmq2 on the untransformed levels; and
  - returns national and regional arithmetic means.
*****************************************************************************/

capture program drop mpi_bootstats_levels
program define mpi_bootstats_levels, rclass

    version 16.0

    syntax [if] [in], ///
        INPUTS(varlist numeric min=1) ///
        OUTPUTS(varlist numeric min=1) ///
        TIME(varname numeric) ///
        DRAWID(varname numeric) ///
        REGION(varname numeric) ///
        REGIONS(numlist)

    marksample touse, novarlist

    preserve

        quietly keep if `touse'

        * idcluster() values can repeat across strata; combine them with the
        * region code to create a globally unique bootstrap panel identifier.
        tempvar boot_panel
        quietly egen long `boot_panel' = group(`region' `drawid')

        tempfile mpi_result

        capture quietly mpi_compute_levels, ///
            inputs(`inputs') ///
            outputs(`outputs') ///
            id(`boot_panel') ///
            time(`time') ///
            dmu(`region') ///
            saving("`mpi_result'")

        if _rc {
            local rc = _rc
            restore
            exit `rc'
        }

        quietly use "`mpi_result'", clear

        * Confirm positive, complete results.
        quietly count if ///
            missing(TFPCH, TECH, TECCH, SECH) | ///
            TFPCH <= 0 | TECH <= 0 | TECCH <= 0 | SECH <= 0

        if r(N) > 0 {
            restore
            exit 498
        }

        * Check the FGNZ decomposition identity up to relative numerical
        * tolerance: TFPCH = TECH x SECH x TECCH.
        quietly count if ///
            abs(TFPCH - TECH*SECH*TECCH) > ///
            1e-5*max(1, abs(TFPCH))

        if r(N) > 0 {
            restore
            exit 498
        }

        foreach index in TFPCH TECH TECCH SECH {

            local idx = lower("`index'")

            * National arithmetic mean.
            quietly summarize `index', meanonly
            local mean_`idx' = r(mean)

            * Region-specific arithmetic means.
            foreach g of numlist `regions' {

                quietly summarize `index' if `region' == `g', meanonly

                if r(N) == 0 {
                    restore
                    exit 498
                }

                local mean_`idx'_r`g' = r(mean)
            }
        }

    restore

    foreach index in tfpch tech tecch sech {

        return scalar mean_`index' = `mean_`index''

        foreach g of numlist `regions' {
            return scalar mean_`index'_r`g' = `mean_`index'_r`g''
        }
    }

end


/*****************************************************************************
6. RUN COUNTY-PANEL BOOTSTRAPS AND DISPLAY 95% CONFIDENCE INTERVALS
*****************************************************************************/

use `analysis_panel', clear

* Initialize the replacement panel identifier. During bootstrap, idcluster()
* assigns distinct IDs to repeated draws of the same original county.
capture drop draw_id
generate long draw_id = county_id
xtset draw_id year

* Build the result list: arithmetic means for four indices, nationally and by
* region. These correspond to the means reported in the submitted tabstat
* tables.
local bslist ""

foreach index in tfpch tech tecch sech {

    local bslist "`bslist' mean_`index'=r(mean_`index')"

    foreach g of numlist `regions' {
        local bslist ///
            "`bslist' mean_`index'_r`g'=r(mean_`index'_r`g')"
    }
}

capture log close _all

log using ///
    "`outdir'/bootstrap_95ci_levels_means.log", ///
    text replace name(bootlog)

foreach model of local models {

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
        local outputs "`hale_output' `child_output' `maternal_output'"
    }
    else {
        display as error "Unknown model: `model'"
        exit 198
    }

    display as text _newline(2) ///
        "Bootstrap 95% CIs using untransformed levels: `model'"

    * The same seed aligns county resampling patterns across outcome models.
    bootstrap `bslist', ///
        reps(`B') ///
        seed(`seed') ///
        level(95) ///
        cluster(county_id) ///
        strata(region_id) ///
        idcluster(draw_id) ///
        saving( ///
            "`outdir'/bootstrap_draws_`model'_levels_means.dta", ///
            replace ///
        ) ///
        nowarn ///
        nodots: ///
        mpi_bootstats_levels, ///
            inputs(`inputs') ///
            outputs(`outputs') ///
            time(year) ///
            drawid(draw_id) ///
            region(region_id) ///
            regions(`regions')

    * Display normal, percentile, and generic bias-corrected 95% intervals.
    * The percentile interval is a clear primary reporting choice.
    estat bootstrap, all

    estimates save ///
        "`outdir'/bootstrap_`model'_levels_means.ster", ///
        replace
}

log close bootlog


/*****************************************************************************
7. COUNTY RESULTS, REGIONAL SUMMARIES, AND CSV EXPORTS
*****************************************************************************/

foreach model of local models {

    use `analysis_panel', clear

    * One observation per county is needed for the single 2014-2022 change.
    keep if year == 2014

    keep ///
        county_id ///
        county ///
        Region ///
        Region_analysis ///
        region_id ///
        water ///
        women_education ///
        marginalizationstatus ///
        electricity ///
        pop_dev ///
        `workforce'

    merge 1:1 county_id using ///
        "`outdir'/productivity_`model'_levels.dta", ///
        assert(match) nogen

    * Preserve full numeric precision; formatting affects display only.
    format TFPCH TECH TECCH SECH %12.6f

    sort Region_analysis county

    save ///
        "`outdir'/county_results_`model'_levels.dta", ///
        replace

    export delimited using ///
        "`outdir'/county_results_`model'_levels.csv", ///
        replace

    display as text _newline ///
        "Regional arithmetic summaries: `model'"

    tabstat TFPCH TECH TECCH SECH, ///
        by(Region_analysis) ///
        statistics(mean sd min max count) ///
        columns(statistics)

    * Save a region-level arithmetic-mean dataset.
    preserve

        collapse ///
            (count) N=TFPCH ///
            (mean) mean_TFPCH=TFPCH ///
                   mean_TECH=TECH ///
                   mean_TECCH=TECCH ///
                   mean_SECH=SECH ///
            (sd)   sd_TFPCH=TFPCH ///
                   sd_TECH=TECH ///
                   sd_TECCH=TECCH ///
                   sd_SECH=SECH, ///
            by(region_id Region_analysis)

        sort region_id

        save ///
            "`outdir'/regional_summary_`model'_levels.dta", ///
            replace

        export delimited using ///
            "`outdir'/regional_summary_`model'_levels.csv", ///
            replace

    restore
}


/*****************************************************************************
8. OPTIONAL SECOND-STAGE REGRESSIONS

These regressions use 2014 covariates as baseline predictors of 2014-2022
productivity change. They do not solve the generated-dependent-variable issue
from first-stage DEA estimation, and conventional clustered standard errors are
fragile with few regional clusters. They are disabled by default.
*****************************************************************************/

local run_regressions 0

if `run_regressions' == 1 {

    capture which esttab
    if _rc {
        display as error "Install esttab first by running: ssc install estout"
        exit 199
    }

    foreach model of local models {

        use ///
            "`outdir'/county_results_`model'_levels.dta", ///
            clear

        estimates clear

        regress TFPCH ///
            water women_education marginalizationstatus electricity, ///
            vce(cluster region_id)
        estimates store model1

        regress TECH ///
            water women_education marginalizationstatus electricity, ///
            vce(cluster region_id)
        estimates store model2

        regress TECCH ///
            water women_education marginalizationstatus electricity, ///
            vce(cluster region_id)
        estimates store model3

        regress SECH ///
            water women_education marginalizationstatus electricity, ///
            vce(cluster region_id)
        estimates store model4

        esttab model1 model2 model3 model4 using ///
            "`outdir'/regression_`model'_levels.rtf", ///
            replace ///
            b(3) ///
            se(3) ///
            label ///
            title( ///
                "Baseline correlates of 2014-2022 productivity change: `model'" ///
            )
    }
}


/*****************************************************************************
9. PRESENTATION AND MAPPING RULES FOR THE REMAINING ANALYSIS
*****************************************************************************/

* Do not round TFPCH, TECH, TECCH, or SECH before classifying values relative
* to 1. Use format for display and the original variables for all calculations.

* Do not replace large estimates with an arbitrary maximum such as 2.5.
* An axis limit may be used in a graph, but underlying values must remain intact.

* Do not use merge ..., force. Use a validated numeric or name crosswalk and
* require matched observations with assert(match).

* The original mapping code alternates between id(county) and id(count). The
* final CSV has no count variable. The polygon identifier must be verified by
* county name before mapping.

* Give child, maternal, HALE, and combined maps unique output filenames so that
* later models do not overwrite earlier files.

* Quadrant classification example using unrounded values:
*
*   count if TFPCH >= 1 & pop_dev <= 100
*   count if TFPCH >= 1 & pop_dev >  100
*   count if TFPCH <  1 & pop_dev <= 100
*   count if TFPCH <  1 & pop_dev >  100


/*****************************************************************************
END OF FILE
*****************************************************************************/
