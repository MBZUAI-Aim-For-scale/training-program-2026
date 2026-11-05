"""Figure browsing keeps decoding and original-file reads bounded."""

import ast
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import streamlit
from PIL import Image

SOURCE = Path("ui/pages/short_medium_benchmarking.py")


def load_functions(names, namespace):
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace


class FigureLoadingTests(unittest.TestCase):
    def test_gallery_only_renders_selected_page(self):
        st = MagicMock()
        st.selectbox.return_value = 2
        st.columns.return_value = [MagicMock(), MagicMock()]
        renderer = MagicMock()
        ns = load_functions({"render_reference_figure_gallery"},
                            {"st": st, "render_benchmark_figure_with_download": renderer})
        paths = [f"figure_{index}.png" for index in range(20)]
        for mode in ("Grid", "Full Width"):
            with self.subTest(mode=mode):
                renderer.reset_mock()
                ns["render_reference_figure_gallery"](paths, mode, "test")
                self.assertEqual([call.args[0] for call in renderer.call_args_list], paths[6:12])
        st.selectbox.return_value = 4
        renderer.reset_mock()
        ns["render_reference_figure_gallery"](paths, "Grid", "test")
        self.assertEqual([call.args[0] for call in renderer.call_args_list], paths[18:20])

    def test_download_reads_original_only_when_clicked(self):
        st = MagicMock()
        st.columns.return_value = [MagicMock(), MagicMock()]
        preview = MagicMock(return_value=b"small preview")
        ns = load_functions({"render_benchmark_figure_with_download"}, {
            "st": st, "Path": Path, "_benchmark_figure_preview": preview,
            "format_figure_label": lambda path: path,
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "figure.png"
            path.write_bytes(b"original PNG")
            with patch.object(Path, "read_bytes", autospec=True, return_value=b"original PNG") as read:
                ns["render_benchmark_figure_with_download"](str(path), enable_zoom=False)
                read.assert_not_called()
                download = st.download_button.call_args.kwargs
                self.assertTrue(callable(download["data"]))
                self.assertEqual(download["data"](), b"original PNG")
                read.assert_called_once()
                self.assertEqual(download["on_click"], "ignore")
            st.image.assert_called_once_with(b"small preview", width="stretch")
            st.warning.assert_not_called()

    def test_preview_is_bounded_cached_and_invalidated_by_metadata(self):
        ns = load_functions({"_benchmark_figure_preview"}, {
            "st": streamlit, "Image": Image, "io": io, "__name__": __name__,
        })
        preview = ns["_benchmark_figure_preview"]
        self.addCleanup(preview.clear)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "figure.png"
            Image.new("RGB", (2400, 1800), "red").save(path)
            stat = path.stat()
            with patch.object(Image, "open", wraps=Image.open) as image_open:
                first = preview(str(path), stat.st_mtime_ns, stat.st_size)
                self.assertEqual(preview(str(path), stat.st_mtime_ns, stat.st_size), first)
                self.assertEqual(image_open.call_count, 1)
            with Image.open(io.BytesIO(first)) as image:
                self.assertLessEqual(image.width, 1600)
                self.assertLessEqual(image.height, 1200)
            Image.new("RGB", (2400, 1800), "blue").save(path)
            self.assertNotEqual(preview(str(path), stat.st_mtime_ns + 1, path.stat().st_size), first)
            with Image.open(path) as original:
                self.assertEqual(original.size, (2400, 1800))

    def test_disabled_netcdf_explorer_does_not_scan_or_open_files(self):
        tree = ast.parse(SOURCE.read_text())
        guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Attribute)
                     and node.test.func.attr == "checkbox"
                     and any(keyword.arg == "key" and isinstance(keyword.value, ast.Constant)
                             and keyword.value.value == "load_nc_explorer" for keyword in node.test.keywords))
        st = MagicMock()
        st.checkbox.return_value = False
        directory = MagicMock()
        open_dataset = MagicMock()
        ns = {"st": st, "ROMP_DEMO_OUT_DIR": directory, "_safe_open_dataset": open_dataset}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), "exec"), ns)  # noqa: S102
        directory.rglob.assert_not_called()
        open_dataset.assert_not_called()
