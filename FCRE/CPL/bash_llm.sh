for t in FewRel Tacred
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
                        CUDA_VISIBLE_DEVICES=0 python train_llm.py \
                            --task_name $t \
                            --num_k 5 \
                            --num_gen 5 \
                            --mixup \
                            --mixup_loss_1 $l \
                            --mixup_loss_2 $m \
                            --rho $r \
                            --SAM \
                            --SAM_type current \
                            --epoch $i \
                            --epoch_mem $j
                    done
                        
                done
            done
        done
    done
done