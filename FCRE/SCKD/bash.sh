for t in FewRel tacred
do
    for i in 8 10
    do
        for j in 6 10
        do
            for l in 0.25 0.5
            do
                for m in 0.25 0.5
                do
                    for r in 0.1 0.05
                    do
                        CUDA_VISIBLE_DEVICES=0 python main.py \
                            --task $t \
                            --shot 5 \
                            --mixup \
                            --loss1_factor $l \
                            --loss2_factor $m \
                            --rho $r \
                            --SAM \
                            --SAM_type current \
                            --step1_epochs $i \
                            --step_2_epochs $j
                            --step_2_epochs 10
                    done
                        
                done
            done
        done
    done
done