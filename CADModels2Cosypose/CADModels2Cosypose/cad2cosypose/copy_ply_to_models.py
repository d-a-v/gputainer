import shutil
from glob import glob
from pathlib import Path

from cad2cosypose import read_config_file

# Import env variables
cfg = read_config_file()

src_ply_paths = glob(
    (Path(cfg["custom_cad_models_path"]) / "**/*.ply").as_posix())

for src_path in src_ply_paths:
    dest_path = Path(cfg["output_dir"]) / "models" / Path(src_path).name
    shutil.copy(src_path, dest_path)
