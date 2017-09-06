max="$1"
count=`expr $max - 1`

export PYTHONPATH=$(pwd)

echo 0 to "$count"
sleep 5

for j in `seq 1 6`
do
    for i in `seq 0 $count`
    do
        python3 asyncdqn/DQN-Distributed.py --slaves_per_url="$max" --urls=localhost --task_index="$i" --alg="$j" &
        echo python3 asyncdqn/DQN-Distributed.py --task_max="$max" --task_index="$i"
        sleep 5
    done

    wait

    rm -rf model_dist train_* frames

    sleep 100
done
