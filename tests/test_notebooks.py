import unittest
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class TestNotebooks(unittest.TestCase):
    def _run_notebook(self, relative_path: str, timeout: int = 600):
        repo_root = Path(__file__).resolve().parents[1]
        nb_path = repo_root / relative_path

        nb = nbformat.read(nb_path, as_version=4)
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3")

        try:
            client.execute()
        except CellExecutionError as e:
            self.fail(f"Notebook {relative_path} failed to execute: {e}")

    def test_tutorial_embedded_complex_executes(self):
        self._run_notebook("doc_source/notebooks/Tutorial-EmbeddedComplex.ipynb")


if __name__ == "__main__":
    unittest.main()
