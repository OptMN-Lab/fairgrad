mkdir -p ./save
mkdir -p ./trainlogs

method=fairgrad
alpha=2.0
seed=0

nohup python -u trainer.py --method=$method --alpha=$alpha --seed=$seed --scale-y=True > trainlogs/$method-alpha$alpha-sd$seed.log 2>&1 &
