import os
import tempfile
from pathlib import Path

import pandas as pd

from cdt.utils.R import launch_R_script

from utils import adj2order, np_to_csv


def cam_pruning(A, X, cutoff, only_pruning=True):
    """Invoke R-based CAM pruning and return the pruned DAG."""
    pruning_path = Path(__file__).resolve().parent / "pruning_R_files/cam_pruning.R"
    with tempfile.TemporaryDirectory() as save_path:
        data_np = X
        data_csv_path = np_to_csv(data_np, save_path)
        dag_csv_path = np_to_csv(A, save_path)

        arguments = dict()
        arguments["{PATH_DATA}"] = data_csv_path
        arguments["{PATH_DAG}"] = dag_csv_path
        arguments["{PATH_RESULTS}"] = os.path.join(save_path, "results.csv")
        arguments["{ADJFULL_RESULTS}"] = os.path.join(save_path, "adjfull.csv")
        arguments["{CUTOFF}"] = str(cutoff)
        arguments["{VERBOSE}"] = "FALSE"  # TRUE, FALSE

        def retrieve_result():
            result = pd.read_csv(arguments["{PATH_RESULTS}"]).values
            os.remove(arguments["{PATH_RESULTS}"])
            os.remove(arguments["{PATH_DATA}"])
            os.remove(arguments["{PATH_DAG}"])
            return result

        dag = launch_R_script(str(pruning_path), arguments, output_function=retrieve_result)

    return dag, adj2order(dag)
