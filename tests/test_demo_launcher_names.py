"""The launcher's demo numbers and names, their setup ids and their lessons must agree.

Demos 4 and 5 swapped places without moving their code: Demo 4 (ForecastCheck) is setup id
demo5 with lesson 05-*.md, and Demo 5 (onset benchmarking) is setup id demo4 with lesson
04-*.md. These checks catch a card pointing at the wrong container or lesson."""
import ast
import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "docs" / "_episodes"


def demos():
    tree = ast.parse((ROOT / "demos/demo1/dashboard.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "DEMOS" for t in n.targets))
    return ast.literal_eval(node.value)


def front_matter(n: int) -> dict:
    files = sorted(EPISODES.glob(f"{n:02d}-*.md"))
    assert len(files) == 1, files
    m = re.match(r"---\n(.*?)\n---", files[0].read_text(), re.S)
    return dict(re.findall(r'(?m)^(title|order): *"?([^"\n]*)"?', m[1]))


class LauncherNamesTests(unittest.TestCase):
    def test_cards(self):
        cards = {d["n"]: d for d in demos()}
        self.assertEqual([d["n"] for d in demos()], [1, 2, 3, 4, 5, "almanac"])
        expected = {2: ("Climate Data Tools", None), 3: ("Run AI Forecast", "demo3"),
                    4: ("AfriClimate AI ForecastCheck", "demo5"), 5: ("Onset Benchmarking", "demo4")}
        for n, (title, setup) in expected.items():
            self.assertEqual(cards[n]["title"], title)
            self.assertEqual(cards[n].get("setup"), setup)

    def test_setup_ids_are_containers_in_demos_toml(self):
        config = {d["id"]: d for d in tomllib.loads((ROOT / "demos/demo1/demos.toml").read_text())["demo"]}
        cards = {d["n"]: d for d in demos()}
        self.assertEqual(config[cards[4]["setup"]]["container"], "forecastcheck")
        self.assertEqual(config[cards[5]["setup"]]["container"], "ai_weather")
        self.assertTrue(config["demo5"]["title"].startswith("Demo 4 · "))
        self.assertTrue(config["demo4"]["title"].startswith("Demo 5 · "))

    def test_each_card_opens_its_own_lesson(self):
        for d in demos():
            if not isinstance(d["n"], int):
                continue
            info = front_matter(d.get("lesson", d["n"]))
            self.assertEqual(info["title"], f"Demo {d['n']}", d["title"])  # the lesson calls itself that demo
            self.assertEqual(int(info["order"]), d["n"])                    # and the site's menu lists it there

    def test_every_lesson_has_an_order(self):
        # (the site sorts its menu by `order`: one without it breaks the sort)
        for f in EPISODES.glob("*.md"):
            self.assertRegex(f.read_text().split("\n---", 1)[0], r"(?m)^order: \d+$", f.name)


if __name__ == "__main__":
    unittest.main()
