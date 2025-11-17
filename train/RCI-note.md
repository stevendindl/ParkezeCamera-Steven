# GPU cluster notes
[RCI SERVER - University of South Carolina](https://docs.google.com/document/d/1S4kpOkPnQeoAcIlQKFjZHeql1IsC4dw_oFTRXOGuGLI/edit?tab=t.0)

```bash
# Login
ssh -p222 <user>@login.rci.sc.edu
```

- navigate to work for user, make project folder, then upload files from laptop/pc to server. tar file directly for quicker upload

```bash
tar -cd - data | pv | ssh -p 222 <user>@<addres>.edu "tar -xf - -C /work/<user>/<project>
```

- put slurm script in `/work/<user>/<project>/`
- use no-cache for venv (e.g. `pip install --no-cache-dir -r requirements.txt`)
- run .sbatch script

```bash
# Check status
squeue -u $USER
```

```
# Output
/work/<user>/<project>/outputs/<output_id>/

# transfer the tarball
rsync -avP -e "ssh -p 222" <user>>@login.rci.sc.edu:<full_path_to_server_file> <full_path_to_user_local_dir>
rsync -avP -e "ssh -p 222" sdindl@login.rci.sc.edu:/work/sdindl/park-data/outputs/20125095/park_run_20125095.tar.gz /home/steven-dindl/Downloads/
```


