"""Rebuild the V3 additions using supplied data and the verified V2 predictions."""
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def hashes(files):
    output={}
    for p in sorted(set(files)):
        with p.open("rb") as stream:
            output[p.relative_to(ROOT).as_posix()]=hashlib.file_digest(stream,"sha256").hexdigest()
    return output


if __name__=="__main__":
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
    for step in ("acoustic_checks.py","propagation.py","timing.py","transfer.py","figures_v3.py"):
        subprocess.run([sys.executable,str(HERE/step)],cwd=ROOT,env=env,check=True)
    support=list((ROOT/"date").glob("Data*.csv")) + list((ROOT/"date").glob("Supplementary5*.wav"))
    prior=["q1_pairwise","q2_held_out_distances","q2_ranges","q3_response_parameters","q3_held_out_probabilities",
           "q3_timing_validation","q4_scenario_results"]
    support.extend(ROOT/"results"/(n+".csv") for n in prior)
    support.append(ROOT/"第二版实验/results/q1_oof.csv")
    base_scripts = ("analyze_acoustics","analyze_effectiveness","analyze_playback","models_acoustic",
                    "models_effectiveness","models_playback","penguin_data","figures_apmcm")
    code=list(HERE.glob("*.py"))+[ROOT/(n+".py") for n in base_scripts]+[
        ROOT/"utils/plot_style.py", ROOT/"utils/export_figure.py"]
    outputs=list((HERE/"results").glob("*.csv"))+list((HERE/"figures").glob("*.png"))
    record={"python":sys.version.split()[0],"versions":{m:version(m) for m in
            ("numpy","scipy","pandas","scikit-learn","matplotlib","Pillow")},
            "command":"python 第三版实验/reproduce.py","inputs":hashes(support),"code":hashes(code),
            "outputs":hashes(outputs)}
    (HERE/"results/manifest.json").write_text(json.dumps(record,ensure_ascii=False,indent=2))
