REM Activate the desired conda environment
call conda activate lightning

REM Run the train.py script with different parameters

python train.py -name "effnet_1"
python train.py -name "effnet_2"
python train.py -name "effnet_3"
python train.py -name "effnet_4"
python train.py -name "effnet_5"
python train.py -name "effnet_6"
python train.py -name "effnet_7"
python train.py -name "effnet_8"
python train.py -name "effnet_9"
python train.py -name "effnet_10"
python train.py -name "effnet_11"
python train.py -name "effnet_14"
python train.py -name "effnet_15"
python train.py -name "effnet_16"

REM Deactivate the conda environment when done
call conda deactivate