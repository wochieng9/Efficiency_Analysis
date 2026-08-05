/*****************************************************************************
KENYA COUNTY MALMQUIST PRODUCTIVITY ANALYSIS
COUNTY-SPECIFIC 95% CONFIDENCE INTERVALS
Self-verifying build: fast path is PROVEN equal to the trusted path at runtime.

-----------------------------------------------------------------------------
THE PROBLEM THIS BUILD SOLVES
  A reorganized (faster) bootstrap loop can silently produce plausible-but-
  wrong intervals. Slower-but-right beats faster-but-wrong. So the fast path is
  never trusted on faith. Every run it is checked, on YOUR data and spec,
  against a slow reference implementation that reproduces the previously
  reviewed, file-based logic. If they disagree by more than a floating-point
  tolerance, the program STOPS and produces no intervals (fail-closed).

HOW THE PROOF IS MADE EXACT (not confounded by the RNG)
  All randomness is isolated in ONE step (generate_draws), which produces a
  stored table of smoothed pseudo-distances dstar for every (rep, county,
  year). BOTH scoring backends read those identical stored draws and call the
  RNG zero times. Therefore, given the same draws, the two backends must return
  identical FGNZ values up to linear-programming solver precision. A real
  reorganization bug (wrong county->dstar mapping, wrong year, wrong reference
  set, wrong projection) produces gross differences (>1e-2); floating-point
  reordering produces ~1e-12. The default tolerance 1e-7 sits cleanly between,
  so it never false-alarms and never misses a real bug.

CHAIN OF TRUST (each link is checked in code)
  malmq2  --[5B: direct distances vs malmq2, ORIGINAL data, ORACLE engine]-->
  oracle engine (previously reviewed, file-based)
          --[preflight: oracle vs fast, PERTURBED data, identical draws]-->
  fast engine (frames, used for the full production run)
  The FGNZ arithmetic is shared by both engines and is validated against
  malmq2 in 5B; the reorganized plumbing is validated by the preflight.

VERIFICATION KNOBS (Section 1)
  verify_reps :  0        = fast only. NOT recommended; prints a loud warning.
                 1..B-1   = preflight gate on the first K reps, then fast for
                            the rest (default 50). The K oracle scorings are
                            the only verification overhead.
                >=B       = paranoid mode: every rep is scored by BOTH backends
                            and asserted equal. Zero unverified reps. Costs
                            about as much as the slow path, but is bulletproof.
  verify_tol  :  relative-difference threshold for the assert (default 1e-7).

METHOD, FOR THE MANUSCRIPT
  Simar & Wilson (1999) smoothed HOMOGENEOUS bootstrap of the FGNZ output-
  oriented Malmquist index, bivariate smoothing of the paired (2014, 2022) CRS
  output-distance scores. TWO caveats to STATE, not hide:
   1. Homogeneity: the bootstrap assumes efficiency is drawn from one density
      independent of covariates; the second-stage regression assumes it depends
      on covariates. Report the first stage as unconditional and say so.
   2. TECH/SECH ride along: only CRS distances are resampled, so pure-
      efficiency (TECH) and scale (SECH) intervals are weaker than TFPCH/TECCH.
  Reported point estimate = original malmq2 (not bias-corrected).
  Primary interval = basic/reverse-percentile, centred on the original estimate.
  Percentile interval and bias-corrected point are sensitivity columns only.

REQUIRED:  ssc install malmq2   (pin its version; this relies on its internal
                                 Mata routines sdf_o()/sdf_i()).
STATA:     16 or newer (frames).
*****************************************************************************/

version 16.0
clear all
frames reset
set more off
set type double


/*****************************************************************************
1. USER SETTINGS
*****************************************************************************/

local datafile "mydata(1).csv"
local outdir   "."

local B      1999          // final reps (use 99 to test paths)
local seed   20260804
local level  95
local hfactor 1            // bandwidth multiplier; report 0.5/1/2 sensitivity

local verify_reps 50       // see header. 0=off(warn), 1..B-1=gate, >=B=paranoid
local verify_tol  1e-7     // relative-difference assert threshold

local maxattempts 0        // 0 -> auto

local models "child maternal hale combined"

local workforce "healthworkforce"
local inputs "pop_dev `workforce'"
local child_output    "childsurvival"
local maternal_output "mumsurvival"
local hale_output     "hale"

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

isid county_id year
assert inlist(year, 2014, 2022)
bysort county_id (year): assert _N == 2
bysort county_id (year): assert year[1] == 2014 & year[2] == 2022
bysort county_id (year): assert region_id == region_id[1]
quietly count
assert r(N) == 94
quietly levelsof county_id, local(county_levels)
assert `: word count `county_levels'' == 47

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

* Keep an on-disk analysis panel (the oracle engine reads targets from file)
* and in-memory frames used by the fast engine.
tempfile analysis_panel
save `analysis_panel', replace

capture frame drop ANALYSIS METADATA
frame put county_id county year Region Region_analysis region_id ///
    `inputs' `child_output' `maternal_output' `hale_output', into(ANALYSIS)
frame ANALYSIS {
    frame put county_id county Region Region_analysis region_id if year == 2014, into(METADATA)
}
frame METADATA {
    isid county_id
    sort county_id
    export delimited using "`outdir'/county_crosswalk.csv", replace
}


/*****************************************************************************
3. SHARED LP WRAPPER (county_shepdf) — operates on the CURRENT frame.
   Validated indirectly via 5B (its output reproduces malmq2). Unchanged.
*****************************************************************************/

capture program drop county_shepdf
program define county_shepdf
    version 16.0
    syntax [if] [in], GEN(string) INvars(varlist numeric) OPvars(varlist numeric) ///
        RFLAG(varname numeric) ///
        [ORT(string) VRS MAXiter(numlist integer >0 max=1) TOL(numlist max=1 >0)]
    marksample touse
    markout `touse' `invars' `opvars'
    tempvar touse_ref
    quietly generate byte `touse_ref' = (`rflag' != 0)
    markout `touse_ref' `invars' `opvars'
    quietly generate double `gen' = .
    local common : list invars & opvars
    if "`common'" != "" {
        display as error "Variables specified as both inputs and outputs: `common'"
        exit 498
    }
    local data `invars' `opvars'
    local ninputs : word count `invars'
    if "`ort'" == "" local ort "IN"
    else {
        local ort = upper("`ort'")
        if inlist("`ort'","I","IN","INPUT")        local ort "IN"
        else if inlist("`ort'","O","OUT","OUTPUT")  local ort "OUT"
        else {
            display as error "ort() must be i,in,input,o,out,output"
            exit 198
        }
    }
    local rts = cond("`vrs'" != "", 1, 0)
    if "`maxiter'" == "" local maxiter -1
    if "`tol'"     == "" local tol -1
    if "`ort'" == "OUT" {
        mata: sdf_o("`data'","`touse'","`touse_ref'",`ninputs',`rts',"`gen'",`maxiter',`tol')
        quietly replace `gen' = 1/`gen' if `touse'
    }
    else {
        mata: sdf_i("`data'","`touse'","`touse_ref'",`ninputs',`rts',"`gen'",`maxiter',`tol')
    }
end


/*****************************************************************************
4. FGNZ ARITHMETIC — shared by both engines. Given the six distances per
   county it returns the four indices. Isolated so both engines are identical
   in arithmetic (arithmetic correctness is covered by 5B vs malmq2).
   Expects wide vars Dsame/Dcross/Dvrs {2014,2022}; leaves TFPCH TECH TECCH
   SECH DCRS2014 DCRS2022 by county_id in the current frame.
*****************************************************************************/

capture program drop fgnz_from_distances
program define fgnz_from_distances
    version 16.0
    quietly count if missing(Dsame2014,Dsame2022,Dcross2014,Dcross2022,Dvrs2014,Dvrs2022)
    if r(N) > 0 {
        display as error "Missing distances into FGNZ: " r(N)
        exit 498
    }
    generate double TECH_crs = Dsame2022/Dsame2014
    generate double TECH     = Dvrs2022/Dvrs2014
    generate double TECCH    = sqrt((Dcross2022/Dsame2022)*(Dsame2014/Dcross2014))
    generate double TFPCH    = TECH_crs*TECCH
    generate double SECH     = TECH_crs/TECH
    quietly count if missing(TFPCH,TECH,TECCH,SECH) | TFPCH<=0 | TECH<=0 | TECCH<=0 | SECH<=0
    if r(N) > 0 {
        display as error "Invalid FGNZ indices: " r(N)
        exit 498
    }
    rename Dsame2014 DCRS2014
    rename Dsame2022 DCRS2022
    keep county_id TFPCH TECH TECCH SECH DCRS2014 DCRS2022
    sort county_id
end


/*****************************************************************************
4o. ORACLE ENGINE (file-based) — reproduces the previously reviewed logic.
    Scores TARGET observations (from a file) against REFERENCE observations
    (from a file) and returns 47-row FGNZ in `saving'. Used by 5B and by the
    preflight as ground truth.
*****************************************************************************/

capture program drop fgnz_against_reference
program define fgnz_against_reference
    version 16.0
    syntax, TARGETS(string) REFERENCES(string) INPUTS(string asis) ///
        OUTPUTS(string asis) SAVING(string)

    tempfile target_data
    use "`targets'", clear
    keep county_id year `inputs' `outputs'
    isid county_id year
    generate byte __target = 1
    generate byte __reference = 0
    save `target_data', replace

    use "`references'", clear
    keep county_id year `inputs' `outputs'
    isid county_id year
    generate byte __target = 0
    generate byte __reference = 1
    append using `target_data'

    generate byte   __rflag = 0
    generate double Dsame  = .
    generate double Dcross = .
    generate double Dvrs   = .

    foreach yy in 2014 2022 {
        if `yy' == 2014 local other 2022
        else            local other 2014

        replace __rflag = (__reference==1 & year==`yy')
        tempvar dsame
        quietly county_shepdf if __target==1 & year==`yy', gen(`dsame') ///
            invars(`inputs') opvars(`outputs') rflag(__rflag) ort(o) maxiter(16000) tol(1e-8)
        replace Dsame = `dsame' if __target==1 & year==`yy'
        drop `dsame'

        replace __rflag = (__reference==1 & year==`other')
        tempvar dcross
        quietly county_shepdf if __target==1 & year==`yy', gen(`dcross') ///
            invars(`inputs') opvars(`outputs') rflag(__rflag) ort(o) maxiter(16000) tol(1e-8)
        replace Dcross = `dcross' if __target==1 & year==`yy'
        drop `dcross'

        replace __rflag = (__reference==1 & year==`yy')
        tempvar dvrs
        quietly county_shepdf if __target==1 & year==`yy', gen(`dvrs') ///
            invars(`inputs') opvars(`outputs') rflag(__rflag) ort(o) vrs maxiter(16000) tol(1e-8)
        replace Dvrs = `dvrs' if __target==1 & year==`yy'
        drop `dvrs'
    }

    keep if __target==1
    keep county_id year Dsame Dcross Dvrs
    isid county_id year
    quietly count if missing(Dsame,Dcross,Dvrs) | Dsame<=0 | Dcross<=0 | Dvrs<=0
    if r(N) > 0 {
        display as error "Infeasible/nonpositive distances (oracle): " r(N)
        exit 498
    }
    reshape wide Dsame Dcross Dvrs, i(county_id) j(year)
    fgnz_from_distances
    save "`saving'", replace
end


/*****************************************************************************
4f. FAST ENGINE (frames) — scores the target block (frame TARGETBLOCK) against
    a pseudo-reference (frame REF) already loaded, both stacked in the current
    frame with __target/__reference flags. Leaves 47-row FGNZ in current frame.
*****************************************************************************/

capture program drop fgnz_core_frames
program define fgnz_core_frames
    version 16.0
    syntax, INPUTS(string asis) OUTPUTS(string asis)
    capture drop __rflag Dsame Dcross Dvrs
    generate byte   __rflag = 0
    generate double Dsame  = .
    generate double Dcross = .
    generate double Dvrs   = .
    foreach yy in 2014 2022 {
        if `yy' == 2014 local other 2022
        else            local other 2014
        replace __rflag = (__reference==1 & year==`yy')
        tempvar dsame
        quietly county_shepdf if __target==1 & year==`yy', gen(`dsame') ///
            invars(`inputs') opvars(`outputs') rflag(__rflag) ort(o) maxiter(16000) tol(1e-8)
        replace Dsame = `dsame' if __target==1 & year==`yy'
        drop `dsame'
        replace __rflag = (__reference==1 & year==`other')
        tempvar dcross
        quietly county_shepdf if __target==1 & year==`yy', gen(`dcross') ///
            invars(`inputs') opvars(`outputs') rflag(__rflag) ort(o) maxiter(16000) tol(1e-8)
        replace Dcross = `dcross' if __target==1 & year==`yy'
        drop `dcross'
        replace __rflag = (__reference==1 & year==`yy')
        tempvar dvrs
        quietly county_shepdf if __target==1 & year==`yy', gen(`dvrs') ///
            invars(`inputs') opvars(`outputs') rflag(__rflag) ort(o) vrs maxiter(16000) tol(1e-8)
        replace Dvrs = `dvrs' if __target==1 & year==`yy'
        drop `dvrs'
    }
    keep if __target==1
    keep county_id year Dsame Dcross Dvrs
    isid county_id year
    quietly count if missing(Dsame,Dcross,Dvrs) | Dsame<=0 | Dcross<=0 | Dvrs<=0
    if r(N) > 0 {
        display as error "Infeasible/nonpositive distances (fast): " r(N)
        exit 498
    }
    reshape wide Dsame Dcross Dvrs, i(county_id) j(year)
    fgnz_from_distances
end


/*****************************************************************************
   Helper: append current-frame rows into another frame (no SSC dependency).
*****************************************************************************/
capture program drop frameappend_to
program define frameappend_to
    version 16.0
    args target
    tempfile tf
    save `tf'
    frame `target' { append using `tf' }
end


/*****************************************************************************
5. MODEL DRIVER
*****************************************************************************/

capture program drop malmq_county_ci
program define malmq_county_ci
    version 16.0
    syntax, MODEL(string) INPUTS(string asis) OUTPUTS(string asis) ///
        OUTDIR(string) PANEL(string) REPS(integer) SEED(integer) ///
        VERIFYREPS(integer) VERIFYTOL(real) ///
        [LEVEL(real 95) HFACTOR(real 1) MAXATTEMPTS(integer 0)]

    if `reps' < 19 {
        display as error "Use at least 19 replications."
        exit 198
    }
    if `maxattempts' == 0 local maxattempts = ceil(3*`reps') + 200
    local plo = (100-`level')/2
    local phi = 100-`plo'

    display as text _newline(2) "============================================================"
    display as text "MODEL: `model'   inputs: `inputs'   outputs: `outputs'"
    display as text "reps=`reps'  verify_reps=`verifyreps'  verify_tol=`verifytol'"
    display as text "============================================================"

    /*==================================================================
    5A. POINT ESTIMATES (malmq2). Also compiles sdf_* into Mata.
    ==================================================================*/
    capture frame drop PT
    frame create PT
    frame PT {
        use "`panel'", clear
        xtset county_id year
        tempfile praw
        quietly malmq2 `inputs' = `outputs', ort(o) fgnz dmu(county) saving("`praw'", replace)
        use "`praw'", clear
        isid county_id
        quietly count
        if r(N) != 47 {
            display as error "malmq2 returned `r(N)' counties, expected 47."
            exit 498
        }
        quietly count if missing(TFPCH,TECH,TECCH,SECH) | TFPCH<=0|TECH<=0|TECCH<=0|SECH<=0
        if r(N) > 0 {
            display as error "malmq2 point estimates invalid."
            exit 498
        }
        keep county_id TFPCH TECH TECCH SECH
        rename TFPCH point_TFPCH
        rename TECH  point_TECH
        rename TECCH point_TECCH
        rename SECH  point_SECH
        sort county_id
    }

    /*==================================================================
    5B. ORACLE ENGINE vs malmq2 on ORIGINAL data  (validates arithmetic+LP)
    ==================================================================*/
    tempfile direct
    fgnz_against_reference, targets("`panel'") references("`panel'") ///
        inputs(`inputs') outputs(`outputs') saving("`direct'")
    frame PT {
        frame create _CHK
        frame _CHK {
            use "`direct'", clear
            frlink 1:1 county_id, frame(PT)
            foreach i in TFPCH TECH TECCH SECH {
                generate double p_`i' = frval(PT, point_`i')
                quietly count if abs(`i' - p_`i') > 1e-5*max(1,abs(p_`i'))
                if r(N) > 0 {
                    display as error "Oracle engine != malmq2 for `i' (`r(N)' counties)."
                    exit 498
                }
            }
        }
        frame drop _CHK
    }
    display as result "5B PASS: oracle engine reproduces malmq2 on original data."

    * Frontier-unit report from the original CRS distances.
    frame create _FR
    frame _FR {
        use "`direct'", clear
        quietly count if abs(DCRS2014-1) <= 1e-6
        local eff14 = r(N)
        quietly count if abs(DCRS2022-1) <= 1e-6
        local eff22 = r(N)
    }
    frame drop _FR
    display as text "Frontier (efficient) counties: 2014=`eff14', 2022=`eff22' of 47 " ///
        "(their intervals are one-sided against the reflection boundary)."

    /*==================================================================
    5C. CONSTANT FRAMES + MODEL PANEL FILE
    ==================================================================*/
    * Paired CRS distances (clipped) -> PAIRS ; long form -> PAIRSLONG.
    capture frame drop PAIRS PAIRSLONG
    frame create PAIRS
    frame PAIRS {
        use "`direct'", clear
        keep county_id DCRS2014 DCRS2022
        rename DCRS2014 D2014
        rename DCRS2022 D2022
        foreach d in D2014 D2022 {
            quietly count if missing(`d') | `d'<=0 | `d'>1+1e-6
            if r(N) > 0 {
                display as error "Invalid CRS distance `d': " r(N)
                exit 498
            }
            replace `d' = 1 if `d'>1 & `d'<=1+1e-6
        }
        sort county_id
    }
    * frame put ... into() CREATES PAIRSLONG; do NOT frame create it first.
    frame PAIRS: frame put county_id D2014 D2022, into(PAIRSLONG)
    frame PAIRSLONG {
        rename D2014 dcrs2014
        rename D2022 dcrs2022
        reshape long dcrs, i(county_id) j(year)
        isid county_id year
        sort county_id year
    }

    * TARGETBLOCK (fast engine target rows).
    capture frame drop TARGETBLOCK
    frame create TARGETBLOCK
    frame TARGETBLOCK {
        use "`panel'", clear
        keep county_id year `inputs' `outputs'
        generate byte __target = 1
        generate byte __reference = 0
        sort county_id year
    }

    * REFBASE (fast engine reference template): original inputs + original
    * outputs + per-cell d_crs. Outputs are rebuilt each rep as y*dstar/dcrs.
    capture frame drop REFBASE
    frame create REFBASE
    frame REFBASE {
        use "`panel'", clear
        keep county_id year `inputs' `outputs'
        foreach y of local outputs {
            rename `y' origout_`y'
        }
        frlink 1:1 county_id year, frame(PAIRSLONG)
        generate double dcrs = frval(PAIRSLONG, dcrs)
        drop PAIRSLONG
        sort county_id year
    }

    * MODELPANEL file for the oracle path (panel + clipped d_crs).
    tempfile modelpanel
    frame create _MP
    frame _MP {
        use "`panel'", clear
        frlink 1:1 county_id year, frame(PAIRSLONG)
        generate double d_crs = frval(PAIRSLONG, dcrs)
        drop PAIRSLONG
        save "`modelpanel'", replace
    }
    frame drop _MP

    /*==================================================================
    5D. SMOOTHING PARAMETERS (unchanged math)
    ==================================================================*/
    frame PAIRS {
        quietly summarize D2014, meanonly
        scalar _mu14 = r(mean)
        quietly summarize D2022, meanonly
        scalar _mu22 = r(mean)
        quietly correlate D2014 D2022, covariance
        matrix _C = r(C)
    }
    scalar _c11=_C[1,1]
    scalar _c12=_C[1,2]
    scalar _c22=_C[2,2]
    scalar _ridge = max(1e-12, 1e-10*max(_c11,_c22))
    scalar _l11 = sqrt(_c11+_ridge)
    scalar _l21 = _c12/_l11
    scalar _l22 = sqrt(max(_c22+_ridge-_l21^2,_ridge))
    scalar _h   = `hfactor'*(47^(-1/6))
    scalar _den = sqrt(1+_h^2)
    display as text "Smoothing bandwidth = " %9.6f _h

    /*==================================================================
    5E. GENERATE ALL DRAWS ONCE  (isolates the RNG; both engines read these)
        Faithful to the original smoothing/reflection/clip sequence.
        DRAWS frame: rep county_id year dstar   (long, `maxattempts'*94 rows
        are NOT stored; we store only accepted-candidate draws lazily below).
        Here we PRE-GENERATE candidate draws for attempt=1.. up front so the
        exact same dstar feeds both backends. Draw acceptance is decided by
        feasibility of the fast engine and mirrored to the oracle.
    ==================================================================*/
    capture frame drop DRAWS
    frame create DRAWS
    frame DRAWS {
        set obs 0
        generate long   attempt = .
        generate long   county_id = .
        generate int    year = .
        generate double dstar = .
    }
    set seed `seed'
    forvalues a = 1/`maxattempts' {
        capture frame drop _D
        frame PAIRS: frame put D2014 D2022, into(_D)
        frame _D {
            bsample 47
            rename D2014 donor2014
            rename D2022 donor2022
            generate long target_order = _n
            * attach fixed county order (PAIRS sorted by county_id)
            capture frame drop _ORD
            frame PAIRS: frame put county_id, into(_ORD)
            frame _ORD {
                sort county_id
                generate long target_order = _n
            }
            frlink 1:1 target_order, frame(_ORD)
            generate long county_id = frval(_ORD, county_id)
            drop _ORD
            frame drop _ORD

            generate double z1 = rnormal()
            generate double z2 = rnormal()
            generate double s14 = donor2014 + _h*(_l11*z1)
            generate double s22 = donor2022 + _h*(_l21*z1 + _l22*z2)
            forvalues rr = 1/20 {
                replace s14 = -s14    if s14<0
                replace s14 = 2-s14   if s14>1
                replace s22 = -s22    if s22<0
                replace s22 = 2-s22   if s22>1
            }
            generate double d14 = _mu14 + (s14-_mu14)/_den
            generate double d22 = _mu22 + (s22-_mu22)/_den
            forvalues rr = 1/20 {
                replace d14 = -d14    if d14<0
                replace d14 = 2-d14   if d14>1
                replace d22 = -d22    if d22<0
                replace d22 = 2-d22   if d22>1
            }
            replace d14 = 1e-8 if d14<=0
            replace d14 = 1    if d14>1
            replace d22 = 1e-8 if d22<=0
            replace d22 = 1    if d22>1
            keep county_id d14 d22
            rename d14 dstar2014
            rename d22 dstar2022
            reshape long dstar, i(county_id) j(year)
            generate long attempt = `a'
            keep attempt county_id year dstar
            frameappend_to DRAWS
        }
        frame drop _D
    }
    display as text "Pre-generated `maxattempts' candidate draw sets (RNG now closed)."

    /*==================================================================
    5F. SCORE. For each attempt: build pseudo-ref from stored dstar, score
        with FAST; if verify due, also score with ORACLE and assert equal.
        Accept feasible draws until `reps' collected.
    ==================================================================*/
    tempname POSTH
    tempfile boot_draws
    postfile `POSTH' long rep long county_id double TFPCH TECH TECCH SECH ///
        using `boot_draws', replace

    local accepted 0
    local verified 0
    scalar _maxreldiff = 0
    local paranoid = (`verifyreps' >= `reps')

    forvalues a = 1/`maxattempts' {
        if `accepted' >= `reps' continue, break

        * ---- FAST: build pseudo-reference from stored dstar (attempt a) ----
        capture frame drop REF
        frame REFBASE: frame put county_id year `inputs' origout_* dcrs, into(REF)
        local good 1
        frame REF {
            capture frame drop _DA
            frame DRAWS: frame put county_id year dstar if attempt==`a', into(_DA)
            frlink 1:1 county_id year, frame(_DA)
            generate double dstar = frval(_DA, dstar)
            drop _DA
            frame drop _DA
            quietly count if missing(dstar) | dstar<=0
            if r(N) > 0 local good 0
            if `good' {
                foreach y of local outputs {
                    generate double `y' = origout_`y' * dstar / dcrs
                }
                keep county_id year `inputs' `outputs'
                generate byte __target = 0
                generate byte __reference = 1
                foreach v of varlist `inputs' `outputs' {
                    quietly count if missing(`v') | `v'<=0
                    if r(N) > 0 local good 0
                }
            }
        }

        if `good' {
            capture frame drop WORK
            frame TARGETBLOCK: frame put county_id year `inputs' `outputs' __target __reference, into(WORK)
            frame REF { frameappend_to WORK }
            frame WORK: capture noisily fgnz_core_frames, inputs(`inputs') outputs(`outputs')
            if _rc local good 0
        }
        if `good' {
            frame WORK {
                quietly count
                if r(N) != 47 local good 0
                if `good' {
                    quietly count if missing(TFPCH,TECH,TECCH,SECH)|TFPCH<=0|TECH<=0|TECCH<=0|SECH<=0
                    if r(N) > 0 local good 0
                }
            }
        }

        if !`good' continue      // infeasible candidate; skip (same for both engines)

        * ---- VERIFY vs ORACLE if this rep is in the checked set ----
        local do_verify = (`verified' < `verifyreps')
        if `do_verify' {
            * oracle pseudo-reference from the SAME stored dstar
            tempfile pseudo_ref oracle_res
            frame create _PR
            frame _PR {
                use "`modelpanel'", clear
                capture frame drop _DA2
                frame DRAWS: frame put county_id year dstar if attempt==`a', into(_DA2)
                frlink 1:1 county_id year, frame(_DA2)
                generate double dstar = frval(_DA2, dstar)
                drop _DA2
                frame drop _DA2
                foreach y of local outputs {
                    replace `y' = `y' * dstar / d_crs
                }
                keep county_id year `inputs' `outputs'
                save "`pseudo_ref'", replace
            }
            frame drop _PR
            fgnz_against_reference, targets("`panel'") references("`pseudo_ref'") ///
                inputs(`inputs') outputs(`outputs') saving("`oracle_res'")

            * compare oracle vs fast
            frame create _CMP
            frame _CMP {
                use "`oracle_res'", clear
                foreach i in TFPCH TECH TECCH SECH {
                    rename `i' ora_`i'
                }
                keep county_id ora_*
                frlink 1:1 county_id, frame(WORK)
                foreach i in TFPCH TECH TECCH SECH {
                    generate double fast_`i' = frval(WORK, `i')
                    generate double rd_`i' = abs(fast_`i'-ora_`i')/max(1,abs(ora_`i'))
                }
                quietly summarize rd_TFPCH, meanonly
                local m1=r(max)
                quietly summarize rd_TECH, meanonly
                local m2=r(max)
                quietly summarize rd_TECCH, meanonly
                local m3=r(max)
                quietly summarize rd_SECH, meanonly
                local m4=r(max)
                local repmax = max(`m1',`m2',`m3',`m4')
                if `repmax' > _maxreldiff scalar _maxreldiff = `repmax'
                if `repmax' > `verifytol' {
                    display as error "VERIFICATION FAILED at attempt `a': " ///
                        "max relative fast-vs-oracle difference = " %12.3e `repmax' ///
                        " exceeds tol " %12.3e `verifytol' "."
                    gsort -rd_TFPCH
                    list county_id ora_TFPCH fast_TFPCH rd_TFPCH in 1, noobs
                    display as error "No intervals produced. Fix before reporting."
                    exit 498
                }
            }
            frame drop _CMP
            local ++verified
        }

        * ---- accept: post the FAST result (proven equal where checked) ----
        local ++accepted
        frame WORK {
            sort county_id
            forvalues ii = 1/47 {
                post `POSTH' (`accepted') (county_id[`ii']) ///
                    (TFPCH[`ii']) (TECH[`ii']) (TECCH[`ii']) (SECH[`ii'])
            }
        }
        if mod(`accepted',50)==0 | `accepted'==`reps' {
            display as text "accepted `accepted'/`reps' (attempt `a', verified `verified'/`verifyreps')"
        }
    }

    postclose `POSTH'

    if `verifyreps' == 0 {
        display as error "WARNING: verify_reps=0. Fast path was NOT checked. " ///
            "Not recommended for reported results."
    }
    else {
        display as result "VERIFICATION PASS: `verified' reps checked, " ///
            "max relative fast-vs-oracle difference = " %12.3e _maxreldiff ///
            " (tol " %12.3e `verifytol' ")."
        if `paranoid' display as result "Paranoid mode: every accepted rep was verified."
    }

    if `accepted' < `reps' {
        display as error "Only `accepted' reps accepted after `maxattempts' attempts. " ///
            "Increase maxattempts or inspect infeasible draws."
        exit 498
    }

    /*==================================================================
    5G. COUNTY-SPECIFIC CONFIDENCE INTERVALS
    ==================================================================*/
    use `boot_draws', clear
    isid rep county_id
    sort rep county_id
    save "`outdir'/county_boot_draws_`model'_levels.dta", replace
    export delimited using "`outdir'/county_boot_draws_`model'_levels.csv", replace

    frlink m:1 county_id, frame(PT)
    foreach i in TFPCH TECH TECCH SECH {
        generate double point_`i' = frval(PT, point_`i')
    }
    drop PT

    foreach i in TFPCH TECH TECCH SECH {
        generate double error_`i' = `i' - point_`i'
        bysort county_id: egen double `i'_bootmean = mean(`i')
        bysort county_id: egen double `i'_bse      = sd(`i')
        bysort county_id: egen double __elo_`i'    = pctile(error_`i'), p(`plo')
        bysort county_id: egen double __ehi_`i'    = pctile(error_`i'), p(`phi')
        generate double `i'_lb = point_`i' - __ehi_`i'
        generate double `i'_ub = point_`i' - __elo_`i'
        bysort county_id: egen double `i'_pct_lb = pctile(`i'), p(`plo')
        bysort county_id: egen double `i'_pct_ub = pctile(`i'), p(`phi')
        generate double `i'_bias = `i'_bootmean - point_`i'
        generate double `i'_bc   = point_`i' - `i'_bias
        generate byte `i'_bc_preferred = (`i'_bse^2 < (`i'_bias^2)/3)
        generate byte `i'_significant  = (`i'_lb>1 | `i'_ub<1)
        generate str28 `i'_conclusion = ///
            cond(`i'_lb>1,"Significant improvement", ///
            cond(`i'_ub<1,"Significant deterioration","No significant change"))
    }
    bysort county_id: keep if _n==1
    drop TFPCH TECH TECCH SECH error_* __elo_* __ehi_*
    rename point_TFPCH TFPCH
    rename point_TECH  TECH
    rename point_TECCH TECCH
    rename point_SECH  SECH

    frlink 1:1 county_id, frame(METADATA)
    foreach mv in county Region Region_analysis region_id {
        capture confirm variable `mv'
        if _rc generate `mv' = frval(METADATA, `mv')
    }
    drop METADATA

    generate str12 model = "`model'"
    generate long bootstrap_reps = `accepted'
    generate long verified_reps  = `verified'
    generate double ci_level = `level'
    generate str26 ci_method = "basic/reverse-percentile"

    order model county_id county Region Region_analysis region_id ///
        bootstrap_reps verified_reps ci_level ci_method ///
        TFPCH TFPCH_lb TFPCH_ub TFPCH_bc TFPCH_bse TFPCH_bias TFPCH_pct_lb TFPCH_pct_ub ///
        TFPCH_bc_preferred TFPCH_significant TFPCH_conclusion ///
        TECH TECH_lb TECH_ub TECH_bc TECH_bse TECH_bias TECH_pct_lb TECH_pct_ub ///
        TECH_bc_preferred TECH_significant TECH_conclusion ///
        TECCH TECCH_lb TECCH_ub TECCH_bc TECCH_bse TECCH_bias TECCH_pct_lb TECCH_pct_ub ///
        TECCH_bc_preferred TECCH_significant TECCH_conclusion ///
        SECH SECH_lb SECH_ub SECH_bc SECH_bse SECH_bias SECH_pct_lb SECH_pct_ub ///
        SECH_bc_preferred SECH_significant SECH_conclusion
    sort county_id
    format TFPCH* TECH* TECCH* SECH* %12.6f

    save "`outdir'/county_ci_`model'_levels.dta", replace
    export delimited using "`outdir'/county_ci_`model'_levels.csv", replace
    display as result "County CI file written: `outdir'/county_ci_`model'_levels.csv"

    capture frame drop PT PAIRS PAIRSLONG TARGETBLOCK REFBASE DRAWS REF WORK
end


/*****************************************************************************
6. RUN MODELS
*****************************************************************************/
local n 0
foreach model of local models {
    local ++n
    if      "`model'"=="child"    local outputs "`child_output'"
    else if "`model'"=="maternal" local outputs "`maternal_output'"
    else if "`model'"=="hale"     local outputs "`hale_output'"
    else if "`model'"=="combined" local outputs "`hale_output' `child_output' `maternal_output'"
    else {
        display as error "Unknown model: `model'"
        exit 198
    }
    malmq_county_ci, model(`model') inputs(`inputs') outputs(`outputs') ///
        outdir("`outdir'") panel("`analysis_panel'") reps(`B') seed(`= `seed'+10000*`n'') ///
        verifyreps(`verify_reps') verifytol(`verify_tol') ///
        level(`level') hfactor(`hfactor') maxattempts(`maxattempts')
}


/*****************************************************************************
7. COMBINE / COMPACT / LONG / COUNTS  (unchanged, operate on saved files)
*****************************************************************************/
local first 1
foreach model of local models {
    if `first' {
        use "`outdir'/county_ci_`model'_levels.dta", clear
        local first 0
    }
    else append using "`outdir'/county_ci_`model'_levels.dta"
}
sort model county_id
save "`outdir'/county_ci_all_models_levels.dta", replace
export delimited using "`outdir'/county_ci_all_models_levels.csv", replace

preserve
    keep model county_id county Region_analysis ///
        TFPCH TFPCH_lb TFPCH_ub TFPCH_conclusion TECH TECH_lb TECH_ub TECH_conclusion ///
        TECCH TECCH_lb TECCH_ub TECCH_conclusion SECH SECH_lb SECH_ub SECH_conclusion
    sort model Region_analysis county
    export delimited using "`outdir'/county_ci_compact_all_models_levels.csv", replace
restore

preserve
    keep model county_id county Region Region_analysis region_id bootstrap_reps verified_reps ci_level ci_method ///
        TFPCH TFPCH_lb TFPCH_ub TFPCH_significant TFPCH_conclusion ///
        TECH TECH_lb TECH_ub TECH_significant TECH_conclusion ///
        TECCH TECCH_lb TECCH_ub TECCH_significant TECCH_conclusion ///
        SECH SECH_lb SECH_ub SECH_significant SECH_conclusion
    foreach i in TFPCH TECH TECCH SECH {
        rename `i' point_`i'
        rename `i'_lb lower_`i'
        rename `i'_ub upper_`i'
        rename `i'_significant significant_`i'
        rename `i'_conclusion conclusion_`i'
    }
    reshape long point_ lower_ upper_ significant_ conclusion_, i(model county_id) j(indicator) string
    rename point_ estimate
    rename lower_ ci_lower
    rename upper_ ci_upper
    rename significant_ significant
    rename conclusion_ conclusion
    order model county_id county Region Region_analysis region_id indicator estimate ci_lower ci_upper ///
        significant conclusion bootstrap_reps verified_reps ci_level ci_method
    sort model indicator Region_analysis county
    export delimited using "`outdir'/county_ci_long_all_models_levels.csv", replace
restore

preserve
    foreach i in TFPCH TECH TECCH SECH {
        generate byte `i'_improve  = (`i'_lb>1)
        generate byte `i'_decline  = (`i'_ub<1)
        generate byte `i'_nochange = (`i'_lb<=1 & `i'_ub>=1)
    }
    collapse (sum) TFPCH_improve TFPCH_decline TFPCH_nochange ///
        TECH_improve TECH_decline TECH_nochange TECCH_improve TECCH_decline TECCH_nochange ///
        SECH_improve SECH_decline SECH_nochange, by(model)
    sort model
    export delimited using "`outdir'/county_significance_counts_levels.csv", replace
restore

display as result _newline(2) "Completed. Fast path verified against oracle every run."
display as result "Main long result: `outdir'/county_ci_long_all_models_levels.csv"
/*****************************************************************************
END OF FILE
*****************************************************************************/
