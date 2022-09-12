#module load python/3
#virtualenv venv
#module unload python/3 
source /data/local/sxa1507/env/venv/bin/activate.csh &
/data/local/sxa1507/env/venv/bin/jupyter-notebook --no-browser --port 9999 &


