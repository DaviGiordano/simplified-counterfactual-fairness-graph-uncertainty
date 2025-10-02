# python fit_causal_models.py --model-type linear --output-dir output/adult/med/linear
python fit_causal_models_gen_eval_counterfactuals.py --model-type diffusion --output-dir output/adult/med/diffusion
# python fit_causal_models.py --model-type lgbm --output-dir output/adult/med/lgbm
python fit_causal_models_gen_eval_counterfactuals.py --model-type causalflow --output-dir output/adult/med/causalflow