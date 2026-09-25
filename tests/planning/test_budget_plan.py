"""Behavioral checks; synthetic profile rates are never planning evidence."""

from dataclasses import replace
import json
import subprocess
import sys
import unittest

from scripts.budget_plan import ROOT, WINDOWS, Inputs, make_cell, read_prices

PRICES, _ = read_prices(ROOT / "experiments/planning/prices.json")


def plan(inputs=Inputs(), window="1hr", band="paid", evidence=None, **kwargs):
    return make_cell(window, band, inputs, PRICES, evidence or {}, **kwargs)


def profiles(cell, cloud_factor=10):
    result = {"profiles": []}
    for schedule in cell["comparisons"]:
        for row in schedule["phases"]:
            if not row["profile_binding"]:
                continue
            factor = cloud_factor if row["resource"] != "local" else 1
            result["profiles"].append({"binding": row["profile_binding"], "attempts": [
                {"receipt_sha256": f"{i:064x}", "status": "complete", "productive_seconds": 100,
                 "counts": {unit: 100 * factor for unit in row["profile_binding"]["units"]}}
                for i in range(1, 4)]})
    return result


class PlannerTests(unittest.TestCase):
    def test_eight_cells_fit_caps_and_keep_unknowns(self):
        cells = [plan(window=w, band=b) for w in WINDOWS for b in ("near_zero", "paid")]
        self.assertEqual(len(cells), 8)
        for cell in cells:
            self.assertFalse(cell["execution_authorized"])
            self.assertEqual(cell["selected"], 0)
            self.assertEqual(len(cell["comparisons"]), 2 if cell["band"] == "paid" else 1)
            for schedule in cell["comparisons"]:
                self.assertLessEqual(schedule["scheduled_wall_seconds"], cell["deadline_seconds"] + 1e-8)
                self.assertLessEqual(schedule["total_upper_with_reserve_usd"], cell["cash_cap_usd"] + 1e-8)
                self.assertEqual(schedule["protected_failure_reserve_usd"], .2 * cell["cash_cap_usd"])
                self.assertLessEqual(schedule["local_active_hours"], 48 + 1e-8)
                self.assertIsNone(schedule["strength_prediction"])
                self.assertTrue(all(value is None for value in schedule["volumes"].values()))
                if cell["band"] == "near_zero":
                    self.assertLess(schedule["total_upper_with_reserve_usd"], 10)

    def test_strict_cash_boundary(self):
        for cap in (10, 1000, float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                plan(band="near_zero", cap=cap)
        self.assertTrue(plan(band="near_zero", cap=9.999)["comparisons"][0]["schedule_feasible"])

    def test_setup_cannot_be_scaled_away(self):
        cell = plan(replace(Inputs(), setup_seconds=600), window="10min")
        self.assertTrue(all(s["status"] == "INFEASIBLE" for s in cell["comparisons"]))
        self.assertTrue(all(s["phases"] == [] for s in cell["comparisons"]))
        self.assertEqual(plan(window="10min")["comparisons"][1]["status"], "INFEASIBLE")

    def test_power_and_availability_clip_week(self):
        for inputs in (replace(Inputs(), local_hours=2.5), replace(Inputs(), electricity_low=3, electricity_high=6)):
            schedule = plan(inputs, window="1week", band="near_zero")["comparisons"][0]
            self.assertTrue(schedule["schedule_feasible"])
            self.assertLessEqual(schedule["local_active_hours"], min(inputs.local_hours, 7.992 / inputs.electricity_high) + 1e-8)
        self.assertFalse(plan(replace(Inputs(), local_hours=0))["comparisons"][0]["schedule_feasible"])

    def test_exact_deadline_and_cap(self):
        cell = plan(deadline=1234, cap=12.34)
        self.assertEqual(cell["deadline_seconds"], 1234)
        self.assertEqual(cell["cash_cap_usd"], 12.34)
        self.assertLessEqual(cell["comparisons"][0]["scheduled_wall_seconds"], 1234)

    def test_ancillary_and_host_hours(self):
        cell = plan(window="1day")
        cloud = cell["comparisons"][1]
        self.assertGreater(cloud["host_hours"], cloud["scheduled_wall_seconds"] / 3600)
        names = [row["phase"] for row in cloud["phases"]]
        self.assertLess(names.index("export"), names.index("evaluation_startup_transfer"))
        subtotal = sum(r["cash_usd"][1] for r in cloud["phases"])
        self.assertAlmostEqual(cloud["cash_usd"][1], subtotal + 2 + cloud["tax_usd"][1])
        high_storage = plan(replace(Inputs(), storage_usd=1000))["comparisons"][1]
        self.assertFalse(high_storage["schedule_feasible"])

    def test_rentals_round_each_lease_without_extending_elapsed_time(self):
        cloud = plan()["comparisons"][1]
        cpu = [r for r in cloud["phases"] if r["resource"] == "cpu"]
        gpu = [r for r in cloud["phases"] if r["resource"] == "gpu"]
        # Generation and evaluation rent separate servers for less than an hour.
        self.assertAlmostEqual(sum(r["billed_host_hours"] for r in cpu), 2)
        # Startup + training use 1053.96 seconds: one 18-minute rental.
        self.assertAlmostEqual(sum(r["billed_host_hours"] for r in gpu), .3)
        self.assertAlmostEqual(cloud["cash_usd"][1], 2.2 + (2 * .3 + .3 * 1.2) * 1.1 + .0105)
        self.assertLessEqual(cloud["scheduled_wall_seconds"], 3600 + 1e-8)
        # Concurrent day workers are billed per host, not per shared wall hour.
        day = plan(window="1day")["comparisons"][1]
        self.assertAlmostEqual(sum(r["billed_host_hours"] for r in day["phases"]
                                   if r["resource"] == "cpu"), 34)

    def test_billing_increments_fit_tight_caps_without_using_reserve(self):
        self.assertFalse(plan(cap=4)["comparisons"][1]["schedule_feasible"])
        self.assertTrue(plan(cap=4.1)["comparisons"][1]["schedule_feasible"])
        self.assertFalse(plan(window="1day", cap=4.1)["comparisons"][1]["schedule_feasible"])
        for cap in (5, 8, 10):
            cloud = plan(window="1day", cap=cap)["comparisons"][1]
            self.assertTrue(cloud["schedule_feasible"])
            self.assertLessEqual(cloud["total_upper_with_reserve_usd"], cap + 1e-8)
        inputs = replace(Inputs(), cloud_startup_seconds=0, transfer_seconds=0,
                         storage_usd=0, transfer_usd=0)
        self.assertFalse(plan(inputs, cap=.1)["comparisons"][1]["schedule_feasible"])

    def test_matched_profiles_select_paid_only_for_useful_gain(self):
        inputs = replace(Inputs(), world_identity="world", pipeline_identity="pipeline", config_identity="config")
        cell = plan(inputs, window="1day")
        evidence = profiles(cell)
        result = plan(inputs, window="1day", evidence=evidence)
        self.assertEqual(result["selected"], 1)
        local = result["comparisons"][0]
        generation = next(r for r in local["phases"] if r["phase"] == "generation")
        self.assertEqual(local["volumes"]["accepted_labels"], [int(generation["wall_seconds"])] * 2)
        self.assertIsNone(result["comparisons"][1]["strength_prediction"])
        self.assertEqual(plan(inputs, window="1day", evidence=profiles(cell, 1))["selected"], 0)

    def test_mismatched_evidence_never_transports(self):
        inputs = replace(Inputs(), world_identity="world", pipeline_identity="pipeline", config_identity="config")
        cell = plan(inputs, window="1day")
        for key, value in (("world_identity", "other"), ("pipeline_identity", "other"),
                           ("config_identity", "other"), ("hardware", "other"), ("hosts", 8),
                           ("workers_per_host", 8), ("units", ["engine_sps"]),
                           ("timing_basis", "end_to_end"), ("deployment", {})):
            evidence = profiles(cell)
            for profile in evidence["profiles"]:
                profile["binding"][key] = value
            result = plan(inputs, window="1day", evidence=evidence)
            self.assertEqual(result["selected"], 0)
            self.assertTrue(all(v is None for s in result["comparisons"] for v in s["volumes"].values()), key)
            cell = plan(inputs, window="1day")

    def test_incomplete_failed_and_duplicate_attempts(self):
        inputs = replace(Inputs(), world_identity="world", pipeline_identity="pipeline", config_identity="config")
        evidence = profiles(plan(inputs))
        for profile in evidence["profiles"]:
            profile["attempts"][-1]["status"] = "failed"
        result = plan(inputs, evidence=evidence)
        self.assertTrue(all(v is None for s in result["comparisons"] for v in s["volumes"].values()))
        self.assertTrue(any("failed" in issue for s in result["comparisons"] for issue in s["issues"]))
        evidence["profiles"][0]["attempts"][1]["receipt_sha256"] = evidence["profiles"][0]["attempts"][0]["receipt_sha256"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            plan(inputs, evidence=evidence)

    def test_training_volume_is_limited_by_available_corpus(self):
        inputs = replace(Inputs(), world_identity="world", pipeline_identity="pipeline", config_identity="config")
        evidence = profiles(plan(inputs))
        for profile in evidence["profiles"]:
            if profile["binding"]["phase"] == "training":
                for attempt in profile["attempts"]:
                    attempt["counts"]["training_exposures"] = 100000000
        result = plan(inputs, evidence=evidence)
        for schedule in result["comparisons"]:
            labels = schedule["volumes"]["accepted_training_rows"]
            exposures = schedule["volumes"]["training_exposures"]
            if labels is not None:
                self.assertLessEqual(exposures[1], labels[1] * 10)
        evidence["profiles"] = [p for p in evidence["profiles"] if p["binding"]["phase"] != "generation"]
        result = plan(inputs, evidence=evidence)
        self.assertTrue(all(s["volumes"]["training_exposures"] is None for s in result["comparisons"]))

    def test_input_types_are_checked(self):
        for inputs in (replace(Inputs(), local_workers=1.5), replace(Inputs(), electricity_high=".09")):
            with self.assertRaises(ValueError):
                plan(inputs)
        with self.assertRaises(ValueError):
            plan(evidence={"profiles": [42]})

    def test_search_budget_is_explicit_and_bound(self):
        with self.assertRaises(ValueError):
            plan(replace(Inputs(), inference_mode="search_assisted"))
        cell = plan(replace(Inputs(), inference_mode="search_assisted", inference_search_simulations=64))
        binding = cell["comparisons"][0]["phases"][1]["profile_binding"]
        self.assertEqual(binding["deployment"]["search_simulations"], 64)

    def test_cli_and_reproduction(self):
        cmd = [sys.executable, str(ROOT / "scripts/budget_plan.py")]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        self.assertEqual(len(json.loads(result.stdout)["cells"]), 8)
        invalid = subprocess.run(cmd + ["--cash-cap", "10"], capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)
        subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_plans.py"), "--check"], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
