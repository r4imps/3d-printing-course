"""Hub self-test: the simulated MG400, the joint jog maths and the hub HTTP API (Simulation mode only,
nothing is sent to a real robot). Run: ../../sim/.venv/Scripts/python -m unittest -v test_hub"""
import base64
import csv
import json
import math
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HUB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HUB))
import hub  # noqa: E402
from sim_robot import SimRobot, Z_FLOOR  # noqa: E402


def wait_idle(robot, timeout=10.0):
    t0 = time.time()
    while robot.call("/status")["target"] is not None:
        if time.time() - t0 > timeout:
            raise AssertionError("simulated arm did not arrive")
        time.sleep(0.02)


class SimRobotTest(unittest.TestCase):
    def setUp(self):
        self.r = SimRobot(pose=(300, 0, -50, 33), speed_ratio=100)

    def test_move_arrives(self):
        self.assertTrue(self.r.call("/move", {"x": 320, "y": 40, "z": -80, "r": 40})["ok"])
        wait_idle(self.r)
        self.assertEqual(self.r.call("/status")["pose"], [320, 40, -80, 40])

    def test_jog_adds_to_the_target(self):
        self.r.call("/jog", {"axis": "z", "delta": -5})
        self.r.call("/jog", {"axis": "z", "delta": -5})
        wait_idle(self.r)
        self.assertAlmostEqual(self.r.call("/status")["pose"][2], -60, places=3)

    def test_out_of_reach_is_refused(self):
        reply = self.r.call("/move", {"x": 600, "y": 0, "z": 0})
        self.assertFalse(reply["ok"])
        self.assertIn("out of reach", reply["error"])

    def test_z_floor_clamps(self):
        reply = self.r.call("/move", {"x": 330, "y": 0, "z": -400})
        if reply["ok"]:
            self.assertEqual(reply["clamped"]["z"], Z_FLOOR)

    def test_status_joints_match_pose(self):
        s = self.r.call("/status")
        x, y, z = hub.kin.forward(*s["joints"][:3])
        self.assertLess(math.dist((x, y, z), s["pose"][:3]), 0.01)
        self.assertAlmostEqual(s["joints"][0] + s["joints"][3], s["pose"][3], places=2)

    def test_disabled_robot_does_not_move(self):
        self.r.call("/disable")
        self.assertFalse(self.r.call("/move", {"x": 310, "y": 0, "z": -50})["ok"])

    def test_pump(self):
        self.assertTrue(self.r.call("/pump", {"mode": "suck"})["ok"])
        self.assertEqual(self.r.call("/status")["pump_mode"], "suck")
        self.assertFalse(self.r.call("/pump", {"mode": "vent"})["ok"])


class JointJogTest(unittest.TestCase):
    status = {"pose": [300.0, 0.0, -50.0, 33.0]}

    def test_j1_turns_about_the_base(self):
        t = hub.joint_target(self.status, "j1", 5)
        self.assertAlmostEqual(math.hypot(t["x"], t["y"]), 300, places=2)
        self.assertAlmostEqual(math.degrees(math.atan2(t["y"], t["x"])), 5, places=2)
        self.assertAlmostEqual(t["r"], 38, places=2)      # R = J1 + J4, J4 kept

    def test_j4_turns_only_r(self):
        t = hub.joint_target(self.status, "j4", -3)
        self.assertAlmostEqual(t["x"], 300, places=2)
        self.assertAlmostEqual(t["r"], 30, places=2)

    def test_j2_changes_reach_not_angle(self):
        t = hub.joint_target(self.status, "j2", 2)
        self.assertGreater(math.hypot(t["x"], t["y"]), 300)
        self.assertAlmostEqual(t["y"], 0, places=2)

    def test_limit_is_refused(self):
        with self.assertRaises(ValueError):
            hub.joint_target(self.status, "j3", 200)


class GlassTrackingTest(unittest.TestCase):
    """The 3D view picks a glass up when the pump sucks on it and leaves it where the pump stops."""

    @classmethod
    def setUpClass(cls):
        cls.live = hub.LivePose()
        t0 = time.time()
        while cls.live.twin is None and cls.live.error is None and time.time() - t0 < 120:
            time.sleep(0.2)
        if cls.live.twin is None:
            raise unittest.SkipTest(f"twin not loaded: {cls.live.error}")

    def pose_over(self, gx, gy, gz):
        tw = self.live._twin_mod
        ox, oy = tw.cup_offset(tw.TOOL_R)
        return gx - ox, gy - oy, self.live.twin.robot_z_for_tip(gz + 1.0)

    def test_pick_carry_place(self):
        live = self.live
        live.glasses = {"glass 1": [300.0, 40.0, 10.0], "glass 2": [300.0, -40.0, 10.0]}
        live.carried = None
        x, y, z = self.pose_over(300.0, 40.0, 10.0)
        live.poses(x, y, z + 30, 33, pump="suck")          # sucking in the air picks nothing
        self.assertIsNone(live.carried)
        live.poses(x, y, z, 33, pump="suck")
        self.assertEqual(live.carried, "glass 1")
        live.poses(x + 20, y, z + 50, 33, pump="suck")
        self.assertAlmostEqual(live.glasses["glass 1"][0], 320.0, places=1)
        live.poses(x + 20, y, z + 50, 33, pump="blow")
        self.assertIsNone(live.carried)
        live.poses(x, y, z + 50, 33, pump="off")
        self.assertAlmostEqual(live.glasses["glass 1"][0], 320.0, places=1)   # stays where it was released
        self.assertEqual(live.glasses["glass 2"], [300.0, -40.0, 10.0])


class RefitCsvTest(unittest.TestCase):
    """cycle.save_refit / next_round on a copy of the real docs/refit_test.csv."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HUB.parent / "lab2" / "slot-picker"))
        import cycle
        cls.cycle = cycle
        cls.real = HUB.parent / "lab2" / "docs" / "refit_test.csv"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "refit_test.csv"
        shutil.copyfile(self.real, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def lines(self):
        return self.path.read_text(encoding="utf-8").splitlines()

    def row(self, glass, ok=True, notes=""):
        return {"glass": glass, "notes": notes, **{f: ok for f in self.cycle.FIELDS}}

    def test_round_1_is_kept_and_round_2_fills_the_template(self):
        before = self.lines()
        self.assertEqual(self.cycle.next_round(self.path), 2)
        self.cycle.save_refit(self.path, 2, [self.row(g) for g in (1, 2, 3, 4)])
        after = self.lines()
        self.assertEqual(after[:5], before[:5])          # header and round 1, byte for byte
        self.assertEqual(len(after), len(before))       # filled in place, no extra rows
        self.assertTrue(all(ln.startswith("2,") and ",yes,yes,yes,yes," in ln for ln in after[5:9]))
        self.assertEqual(after[9:], before[9:])
        self.assertEqual(self.cycle.next_round(self.path), 3)

    def test_saving_a_round_again_appends_instead_of_overwriting(self):
        self.cycle.save_refit(self.path, 2, [self.row(1, notes="first")])
        self.cycle.save_refit(self.path, 2, [self.row(1, ok=False, notes="second")])
        with self.path.open(newline="", encoding="utf-8") as f:
            r2g1 = [r for r in csv.DictReader(f) if r["round"] == "2" and r["glass_id"] == "1"]
        self.assertEqual([r["notes"] for r in r2g1], ["first", "second"])
        self.assertEqual(r2g1[1]["placed_at_output"], "no")

    def test_partly_saved_round_stays_the_next_round(self):
        # glass 1 saved, page reloaded: glasses 2-4 still belong to round 2
        self.cycle.save_refit(self.path, 2, [self.row(1)])
        self.assertEqual(self.cycle.next_round(self.path), 2)
        self.cycle.save_refit(self.path, 2, [self.row(g) for g in (2, 3, 4)])
        self.assertEqual(self.cycle.next_round(self.path), 3)


class HubApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        cls.port = s.getsockname()[1]
        s.close()
        handler, cls.robots = hub.build(hub.HERE.parent / "lab2", cls.port)
        cls.server = ThreadingHTTPServer(("127.0.0.1", cls.port), handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.robots.sim.speed_ratio = 30

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def req(self, path, body=None):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                   data=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(r, timeout=10))
        except urllib.error.HTTPError as e:
            return json.load(e)

    def test_page_and_picker_load(self):
        for path in ("/", "/picker"):
            body = urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10).read()
            self.assertIn(b"<html", body.lower()[:400])

    def test_picker_page_is_the_hub_version(self):
        body = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/picker", timeout=10).read().decode()
        self.assertIn('id="log"', body)          # event log instead of the old status block
        self.assertNotIn('id="status"', body)

    def test_state_starts_in_simulation(self):
        s = self.req("/hub/state")
        self.assertEqual(s["mode"], "sim")
        self.assertTrue(s["robot"]["simulation"])
        self.assertEqual(len(s["robot"]["joints"]), 4)

    def test_motion_is_locked_until_armed(self):
        self.req("/hub/arm", {"armed": False})
        r = self.req("/hub/jog", {"space": "cart", "axis": "z", "delta": -1})
        self.assertFalse(r["ok"])
        self.assertIn("Motion is off", r["error"])
        self.assertFalse(self.req("/api/goto", {"cell": "D+2", "hover_z": -110})["ok"])
        self.assertTrue(self.req("/hub/arm", {"armed": True})["armed"])
        self.assertTrue(self.req("/hub/state")["motion"]["armed"])

    def test_mode_switch_locks_motion(self):
        self.req("/hub/arm", {"armed": True})
        self.req("/hub/mode", {"mode": "real"})
        self.req("/hub/mode", {"mode": "sim"})
        self.assertFalse(self.req("/hub/state")["motion"]["armed"])

    def test_cartesian_and_joint_jog(self):
        self.req("/hub/arm", {"armed": True})
        z0 = self.req("/hub/state")["robot"]["pose"][2]
        self.assertTrue(self.req("/hub/jog", {"space": "cart", "axis": "z", "delta": -2})["ok"])
        wait_idle(self.robots.sim)
        self.assertAlmostEqual(self.req("/hub/state")["robot"]["pose"][2], z0 - 2, places=2)
        self.assertTrue(self.req("/hub/jog", {"space": "joint", "axis": "j1", "delta": 3})["ok"])
        wait_idle(self.robots.sim)

    def test_jog_rejects_big_steps_and_bad_axes(self):
        self.req("/hub/arm", {"armed": True})
        self.assertFalse(self.req("/hub/jog", {"space": "cart", "axis": "z", "delta": -50})["ok"])
        self.assertFalse(self.req("/hub/jog", {"space": "joint", "axis": "j9", "delta": 1})["ok"])

    def test_slot_picker_api_runs_on_the_simulation(self):
        lay = self.req("/api/layout")
        self.assertIn("cells", lay)
        st = self.req("/api/state")
        self.assertEqual(st["robot"]["mode_name"], "ENABLED")

    def test_refit_round_2_runs_and_saves_without_reteaching(self):
        # a full round on the simulation with the taught points as they are, saved into a copy of the CSV
        sp = sys.modules["slot_picker"]
        real_csv = sp.REFIT_CSV
        with tempfile.TemporaryDirectory() as tmp:
            sp.REFIT_CSV = Path(tmp) / "refit_test.csv"
            shutil.copyfile(real_csv, sp.REFIT_CSV)
            try:
                lay = self.req("/api/layout")
                self.assertEqual(lay["refit_next_round"], 2)
                self.req("/hub/arm", {"armed": True})
                reply = self.req("/api/cycle/run", {"glasses": [1, 2, 3, 4], "round": lay["refit_next_round"]})
                self.assertTrue(reply["ok"], reply)
                t0 = time.time()
                while self.req("/api/state")["seq"]["busy"]:
                    self.assertLess(time.time() - t0, 240, "simulated round did not finish")
                    time.sleep(0.2)
                seq = self.req("/api/state")["seq"]
                self.assertIsNone(seq["error"])
                results = seq["cycle"]["results"]
                self.assertTrue(all(all(r.values()) for r in results.values()), results)
                rows = [{"glass": int(g), "notes": "sim", **r} for g, r in results.items()]
                self.assertTrue(self.req("/api/refit", {"round": 2, "rows": rows})["ok"])
                self.assertEqual(self.req("/api/layout")["refit_next_round"], 3)
                with sp.REFIT_CSV.open(newline="", encoding="utf-8") as f:
                    saved = list(csv.DictReader(f))
                self.assertEqual(len(saved), 20)
                self.assertTrue(all(r["picked_from_input"] == "yes" for r in saved if r["round"] in ("1", "2")))
            finally:
                sp.REFIT_CSV = real_csv

    def test_mode_switch(self):
        self.assertTrue(self.req("/hub/mode", {"mode": "real"})["ok"])
        self.assertEqual(self.req("/hub/state")["mode"], "real")
        self.assertTrue(self.req("/hub/mode", {"mode": "sim"})["ok"])
        self.assertFalse(self.req("/hub/mode", {"mode": "both"})["ok"])

    def test_webcam_frame_is_saved(self):
        def post(body):
            return self.req("/hub/webcam/frame", {"jpeg": base64.b64encode(body).decode()})
        self.assertFalse(post(b"not a picture")["ok"])
        reply = post(b"\xff\xd8\xff\xe0 fake jpeg")
        self.assertTrue(reply["ok"])
        saved = hub.HERE.parent / "lab2" / "webcam-frames" / reply["name"]
        self.assertTrue(saved.is_file())
        saved.unlink()

    def test_foreign_origin_is_refused(self):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}/hub/mode", data=b'{"mode":"real"}',
                                   headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(r, timeout=5)
        self.assertEqual(e.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
