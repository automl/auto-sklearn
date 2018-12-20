rm commands.txt
for bench in Branin Bohachevsky Camelback Forrester GoldsteinPrice Hartmann3 Hartmann6 Levy Rosenbrock SinOne SinTwo; do
    for seed in {11..32}; do
        for predict_incumbent in True False; do
            for do_hpo in True False; do
                for reinterpolation in False; do
                    for bootstrap in True False; do
                        for new_search in True False; do
                            if [ "$do_hpo" == "False" ] && [ "$reinterpolation" == "True" ]; then
				continue
			    fi
                            if [ "$do_hpo" == "False" ] && [ "$reinterpolation" == "False" ] && [ "$new_search" == "False" ] && [ "$bootstrap" == "True" ]; then
				continue
			    fi
                            #if [ "$reinterpolation" == "False" ] && [ "$bootstrap" == "True" ]; then
                            #    continue
                            #fi
                            cmd="python scripts/run.py  --benchmark $bench --n_parallel 1 --repetitions 1 --smac_path smacs/SMAC/smac3/ --first_n $seed --predict_incumbent $predict_incumbent --do_hpo $do_hpo --reinterpolation $reinterpolation --bootstrap $bootstrap --new_search $new_search"
                            echo $cmd >> commands.txt
                        done
                    done
                done
            done
        done
    done
done
