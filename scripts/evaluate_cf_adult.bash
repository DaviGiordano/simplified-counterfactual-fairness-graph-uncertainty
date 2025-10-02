# python3 run_classifier_pipeline.py --dataset adult --knowledge low --classifier LR
# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier LR

# python3 run_classifier_pipeline.py --dataset adult --knowledge low --classifier GB
# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier GB

# python3 run_classifier_pipeline.py --dataset adult --knowledge low --classifier RF
# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier RF

# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier LR_no_sensitive
# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier GB_no_sensitive
# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier RF_no_sensitive
# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier FAIRGBM
# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier FAIRGBM_equal_opportunity
# python3 run_classifier_pipeline.py --dataset adult --knowledge med --classifier FAIRGBM_predictive_equality
