for d in ACE MAVEN
do
    for i in 5
    do
        for m in 10
        do
            for k in shuffle
            do
                for j in 5 10
                do
                    for p in 0 1 2 3 4
                    do
                        CUDA_VISIBLE_DEVICES=0 python train.py \
                            --data-root ./augmented_data \
                            --stream-root ./augmented_data \
                            --dataset $d \
                            --backbone bert-base-uncased \
                            --lr 2e-5 \
                            --decay 1e-4 \
                            --no-freeze-bert \
                            --shot-num $j \
                            --batch-size 4 \
                            --device cuda:0 \
                            --log \
                            --log-dir ./sam/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                            --tb-dir ./sam/log_tensorboard/02-10-nomap-clreps\
                            --log-name a${k}_lnone_r${i} \
                            --dweight_loss \
                            --rep-aug mean \
                            --distill mul \
                            --epoch 30 \
                            --class-num $m \
                            --single-label \
                            --cl-aug $k \
                            --aug-repeat-times $i \
                            --joint-da-loss none \
                            --sub-max \
                            --cl_temp 0.07 \
                            --tlcl \
                            --ucl \
                            --skip-first-cl ucl+tlcl \
                            --perm-id $p \
                            --aug-dropout-times 0 \
                            --sam \
                            --sam-type current \
                            --rho 0.05
                    done
                done
            done
        done
    done
done