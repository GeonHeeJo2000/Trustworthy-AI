python test/generate_data.py ngsim
python test/generate_data.py apolloscape
python test/generate_data.py ngsim

python test/test.py --dataset apolloscape --model grip --mode single_frame --augment --smooth 0 --blackbox
python test/test.py --dataset dfl --model dualtransformer --mode single_frame --augment --smooth 0
