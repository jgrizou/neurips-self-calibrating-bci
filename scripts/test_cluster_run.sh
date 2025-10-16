for N in {0..20}
do
    echo "Running python script with -N $N"
    python ../src/run_experiments.py -N $N
done