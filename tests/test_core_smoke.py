import numpy as np
import pandas as pd

from efficiency_tool.dea.core import calculate_dea_with_slacks, calculate_scale_diagnostics
from efficiency_tool.sfa.core import calculate_sfa_production
from efficiency_tool.results.tables import build_peer_weights_table, build_target_table


def make_sample_data():
    return pd.DataFrame(
        {
            "DMU_ID": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "Input": [1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 4.5, 5.0],
            "Output": [1.0, 2.2, 2.7, 3.9, 2.8, 3.4, 4.2, 5.1],
            "Original_Row": list(range(1, 9)),
        }
    )


def test_dea_core_smoke():
    data = make_sample_data()
    dea = calculate_dea_with_slacks(data, ["Input"], ["Output"], orientation="output", returns="crs")
    scores = np.asarray(dea["efficiency"], dtype=float)
    assert scores.shape == (len(data),)
    assert np.nanmin(scores) >= 0.0
    assert np.nanmax(scores) <= 1.0


def test_scale_diagnostics_smoke():
    data = make_sample_data()
    scale = calculate_scale_diagnostics(data, ["Input"], ["Output"], orientation="output")
    assert len(scale["scale_efficiency"]) == len(data)
    assert len(scale["returns_to_scale"]) == len(data)


def test_sfa_core_smoke():
    data = make_sample_data()
    sfa = calculate_sfa_production(data, ["Input"], "Output", "Cobb-Douglas", "Production")
    assert len(sfa["efficiency"]) == len(data)


def test_target_table_smoke():
    data = make_sample_data()
    dea = calculate_dea_with_slacks(data, ["Input"], ["Output"])
    targets = build_target_table(data, dea, ["Input"], ["Output"], data["DMU_ID"].tolist())
    assert {"DMU_ID", "DEA_Efficiency", "DEA_Peers"}.issubset(targets.columns)



def test_peer_weights_table_smoke():
    lambdas = np.array([[1.0, 0.0], [0.25, 0.75]])
    peers = build_peer_weights_table(lambdas, ["A", "B"])
    assert {"DMU_ID", "Peer_DMU_ID", "Lambda_Weight"}.issubset(peers.columns)
    assert len(peers) == 3

from efficiency_tool.utils.data import clean_and_prepare_data


def test_clean_data_uses_time_column_in_dmu_id():
    data = pd.DataFrame(
        {
            "Facility": ["A", "A", "B", "B"],
            "Year": [2022, 2023, 2022, 2023],
            "Input": [10.0, 11.0, 20.0, 21.0],
            "Output": [100.0, 105.0, 190.0, 200.0],
        }
    )
    clean, summary = clean_and_prepare_data(
        data,
        input_cols=["Input"],
        output_cols=["Output"],
        id_col="Facility",
        time_col="Year",
        group_col=None,
        env_cols=[],
    )

    assert clean["DMU_ID"].tolist() == ["A | 2022", "A | 2023", "B | 2022", "B | 2023"]
    assert summary["duplicate_ids"] == 0
    assert summary["dmu_id_basis"] == "Facility + Year"


def test_clean_data_suffixes_true_duplicate_dmu_periods():
    data = pd.DataFrame(
        {
            "Facility": ["A", "A", "A"],
            "Year": [2022, 2022, 2023],
            "Input": [10.0, 12.0, 11.0],
            "Output": [100.0, 110.0, 105.0],
        }
    )
    clean, summary = clean_and_prepare_data(
        data,
        input_cols=["Input"],
        output_cols=["Output"],
        id_col="Facility",
        time_col="Year",
        group_col=None,
        env_cols=[],
    )

    assert summary["duplicate_ids"] == 1
    assert clean["DMU_ID"].is_unique
    assert clean["DMU_ID"].tolist() == ["A | 2022 (row 1)", "A | 2022 (row 2)", "A | 2023"]

from efficiency_tool.dea.core import calculate_dea_with_slacks_by_group
from efficiency_tool.results.tables import build_results_table


def test_sfa_already_logged_variables_are_not_logged_again():
    data = make_sample_data().copy()
    data["ln_Input"] = np.log(data["Input"])
    data["ln_Output"] = np.log(data["Output"])
    sfa_raw = calculate_sfa_production(data, ["Input"], "Output", "Cobb-Douglas", "Production", data_are_logged=False)
    sfa_logged = calculate_sfa_production(data, ["ln_Input"], "ln_Output", "Cobb-Douglas", "Production", data_are_logged=True)
    assert len(sfa_logged["efficiency"]) == len(data)
    assert sfa_logged["data_are_logged"] is True
    assert "ln_Input" in sfa_logged["param_names"]
    assert np.allclose(sfa_raw["efficiency"], sfa_logged["efficiency"], equal_nan=True, atol=1e-6)


def test_grouped_dea_only_uses_peers_in_same_period():
    data = pd.DataFrame(
        {
            "DMU_ID": ["A | 2022", "B | 2022", "A | 2023", "B | 2023"],
            "Year": [2022, 2022, 2023, 2023],
            "Input": [1.0, 2.0, 1.0, 2.0],
            "Output": [1.0, 1.0, 2.0, 2.0],
            "Original_Row": [1, 2, 3, 4],
        }
    )
    dea = calculate_dea_with_slacks_by_group(data, "Year", ["Input"], ["Output"], orientation="input", returns="vrs")
    lambdas = np.asarray(dea["lambdas"])
    assert lambdas[0, 2] == 0.0
    assert lambdas[1, 3] == 0.0
    assert lambdas[2, 0] == 0.0
    assert lambdas[3, 1] == 0.0


def test_results_table_includes_benchmarking_style_output_score():
    data = make_sample_data()
    dea = calculate_dea_with_slacks(data, ["Input"], ["Output"], orientation="output", returns="crs")
    scale = calculate_scale_diagnostics(data, ["Input"], ["Output"], orientation="output")
    support = pd.DataFrame(
        {
            "DMU_ID": data["DMU_ID"],
            "Reference_Count": 0,
            "Reference_Weight_Sum": 0.0,
            "Benchmark_Support_Score": 0.0,
        }
    )
    sfa = calculate_sfa_production(data, ["Input"], "Output")
    results = build_results_table(
        data,
        dea,
        scale,
        sfa,
        None,
        support,
        None,
        [],
        ["Input"],
        ["Output"],
        dea_orientation="output",
        sfa_input_cols=["Input"],
        sfa_output_col="Output",
    )
    assert "DEA_R_Benchmarking_Style_Score" in results.columns
    assert np.allclose(results["DEA_R_Benchmarking_Style_Score"], results["DEA_Radial_Factor"], equal_nan=True)

from efficiency_tool.results.time_trends import build_time_trend_tables
from efficiency_tool.sfa.core import COST_EFFICIENCY_STATA


def test_time_trend_tables_compare_same_unit_first_to_last():
    data = pd.DataFrame(
        {
            "County": ["A", "A", "B", "B"],
            "Year": [2014, 2022, 2014, 2022],
            "DMU_ID": ["A | 2014", "A | 2022", "B | 2014", "B | 2022"],
            "DEA_Efficiency": [0.70, 0.85, 0.90, 0.80],
            "SFA_Efficiency": [0.60, 0.65, 0.75, 0.78],
            "Scale_Efficiency": [0.95, 0.96, 0.97, 0.98],
        }
    )
    trends = build_time_trend_tables(data, unit_col="County", time_col="Year")
    assert trends["available"] is True
    assert trends["period_labels"] == ["2014", "2022"]

    unit_change = trends["unit_change"]
    a_dea = unit_change[(unit_change["Trend_Unit_ID"] == "A") & (unit_change["Score"] == "DEA_Efficiency")].iloc[0]
    b_dea = unit_change[(unit_change["Trend_Unit_ID"] == "B") & (unit_change["Score"] == "DEA_Efficiency")].iloc[0]
    assert np.isclose(a_dea["Improvement"], 0.15)
    assert bool(a_dea["Improved"])
    assert np.isclose(b_dea["Improvement"], -0.10)
    assert bool(b_dea["Declined"])

    change_summary = trends["change_summary"]
    dea_summary = change_summary[change_summary["Score"] == "DEA_Efficiency"].iloc[0]
    assert np.isclose(dea_summary["Share_Improved"], 0.5)


def test_time_trend_tables_treat_stata_cost_ratio_as_lower_is_better():
    data = pd.DataFrame(
        {
            "Facility": ["A", "A", "B", "B"],
            "Period": [1, 2, 1, 2],
            "DMU_ID": ["A | 1", "A | 2", "B | 1", "B | 2"],
            "DEA_Efficiency": [0.90, 0.95, 0.90, 0.90],
            "SFA_Efficiency": [1.20, 1.10, 1.05, 1.20],
            "SFA_Cost_Efficiency_Convention": [COST_EFFICIENCY_STATA] * 4,
        }
    )
    trends = build_time_trend_tables(data, unit_col="Facility", time_col="Period")
    sfa_change = trends["unit_change"][trends["unit_change"]["Score"] == "SFA_Efficiency"]
    a = sfa_change[sfa_change["Trend_Unit_ID"] == "A"].iloc[0]
    b = sfa_change[sfa_change["Trend_Unit_ID"] == "B"].iloc[0]
    assert np.isclose(a["Raw_Change"], -0.10)
    assert np.isclose(a["Improvement"], 0.10)
    assert bool(a["Improved"])
    assert np.isclose(b["Improvement"], -0.15)
    assert bool(b["Declined"])

from efficiency_tool.dea.malmquist import calculate_malmquist_indices


def test_malmquist_user_example_input_crs_improves_productivity():
    inputs_t1 = np.array([[3, 5], [4, 4], [2, 6]], dtype=float)
    outputs_t1 = np.array([[10], [12], [9]], dtype=float)
    inputs_t2 = np.array([[2.5, 4.5], [3.5, 3.8], [1.8, 5.5]], dtype=float)
    outputs_t2 = np.array([[11], [13], [10]], dtype=float)
    rows = []
    for i in range(3):
        rows.append({"DMU": f"DMU{i + 1}", "Year": 1, "x1": inputs_t1[i, 0], "x2": inputs_t1[i, 1], "y": outputs_t1[i, 0]})
        rows.append({"DMU": f"DMU{i + 1}", "Year": 2, "x1": inputs_t2[i, 0], "x2": inputs_t2[i, 1], "y": outputs_t2[i, 0]})

    result = calculate_malmquist_indices(pd.DataFrame(rows), ["x1", "x2"], ["y"], "DMU", "Year", orientation="input", returns="crs")
    pairwise = result["pairwise"].sort_values("Trend_Unit_ID")

    assert result["available"] is True
    assert np.all(pairwise["Malmquist_MPI"].to_numpy(dtype=float) > 1.0)
    assert np.allclose(
        pairwise["Malmquist_MPI"].to_numpy(dtype=float),
        np.array([1.28246637, 1.17568852, 1.22887357]),
        atol=1e-6,
    )
    assert np.allclose(
        pairwise["Malmquist_MPI"].to_numpy(dtype=float),
        pairwise["Efficiency_Change_EC"].to_numpy(dtype=float) * pairwise["Technical_Change_TC"].to_numpy(dtype=float),
        atol=1e-10,
    )


def test_malmquist_output_orientation_uses_reciprocal_distance_convention():
    data = pd.DataFrame(
        {
            "DMU": ["A", "A", "B", "B"],
            "Year": [1, 2, 1, 2],
            "Input": [2.0, 1.5, 3.0, 2.5],
            "Output": [10.0, 12.0, 12.0, 14.0],
        }
    )
    result = calculate_malmquist_indices(data, ["Input"], ["Output"], "DMU", "Year", orientation="output", returns="crs")
    pairwise = result["pairwise"]
    assert result["available"] is True
    assert np.all(pairwise["Malmquist_MPI"].to_numpy(dtype=float) > 1.0)


def test_malmquist_only_compares_units_present_in_adjacent_periods():
    data = pd.DataFrame(
        {
            "DMU": ["A", "A", "B", "C"],
            "Year": [1, 2, 1, 2],
            "Input": [1.0, 0.9, 2.0, 3.0],
            "Output": [1.0, 1.1, 2.0, 3.0],
        }
    )
    result = calculate_malmquist_indices(data, ["Input"], ["Output"], "DMU", "Year", orientation="input", returns="crs")
    assert result["available"] is True
    assert result["pairwise"]["Trend_Unit_ID"].tolist() == ["A"]
    assert int(result["period_pairs"].iloc[0]["Common_Units_Compared"]) == 1
