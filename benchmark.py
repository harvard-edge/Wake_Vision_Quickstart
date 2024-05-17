
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import pandas as pd
import tensorflow as tf

import pp_ops

def filter_by_benchmark_set(ds_entry, benchmark_name):
    return ds_entry[benchmark_name] == 1
    
    
    
    
def f1(tp_rate, fp_rate, fn_rate):
    return 2 * tp_rate / (2 * tp_rate + fp_rate + fn_rate)

def lighting_eval(model, ds):
    input_shape = model.input_shape[1:]
    lighting_dark_person = ds.filter(lambda x: filter_by_benchmark_set(x, "dark") and filter_by_benchmark_set(x, "person")) 
    lighting_normal_person = ds.filter(lambda x: filter_by_benchmark_set(x, "normal_lighting") and filter_by_benchmark_set(x, "person")) 
    lighting_bright_person = ds.filter(lambda x: filter_by_benchmark_set(x, "bright") and filter_by_benchmark_set(x, "person"))
    
    lighting_dark_non_person = ds.filter(lambda x: filter_by_benchmark_set(x, "dark") and not filter_by_benchmark_set(x, "person"))
    lighting_normal_non_person = ds.filter(lambda x: filter_by_benchmark_set(x, "normal_lighting") and not filter_by_benchmark_set(x, "person"))
    lighting_bright_non_person = ds.filter(lambda x: filter_by_benchmark_set(x, "bright") and not filter_by_benchmark_set(x, "person"))

    person_dark_score = model.evaluate(pp_ops.preprocessing(lighting_dark_person, input_shape=input_shape), verbose=0)
    person_normal_light_score = model.evaluate(pp_ops.preprocessing(lighting_normal_person, input_shape=input_shape), verbose=0)
    person_bright_score = model.evaluate(pp_ops.preprocessing(lighting_bright_person, input_shape=input_shape), verbose=0)
    
    non_person_dark_score = model.evaluate(pp_ops.preprocessing(lighting_dark_non_person, input_shape=input_shape), verbose=0)
    non_person_normal_light_score = model.evaluate(pp_ops.preprocessing(lighting_normal_non_person, input_shape=input_shape), verbose=0)
    non_person_bright_score = model.evaluate(pp_ops.preprocessing(lighting_bright_non_person, input_shape=input_shape), verbose=0)
    
    dark_f1 = f1(person_dark_score[1], 1-non_person_dark_score[1], 1-person_dark_score[1])
    normal_light_f1 = f1(person_normal_light_score[1], 1-non_person_normal_light_score[1], 1-person_normal_light_score[1])
    bright_f1 = f1(person_bright_score[1], 1-non_person_bright_score[1], 1-person_bright_score[1])
    
    result = pd.DataFrame({
        'lighting-dark': [dark_f1],
        'lighting-normal_light': [normal_light_f1],
        'lighting-bright': [bright_f1]})

    print(result)

    return result

def distance_eval(model, ds):
    input_shape = model.input_shape[1:]

    distance_near = ds.filter(lambda x: filter_by_benchmark_set(x, "near"))
    distance_mid = ds.filter(lambda x: filter_by_benchmark_set(x, "medium_distance"))
    distance_far = ds.filter(lambda x: filter_by_benchmark_set(x, "far"))
    no_person = ds.filter(lambda x: not filter_by_benchmark_set(x, "person"))
    

    near_score = model.evaluate(pp_ops.preprocessing(distance_near, input_shape=input_shape), verbose=0)
    mid_score = model.evaluate(pp_ops.preprocessing(distance_mid, input_shape=input_shape), verbose=0)
    far_score = model.evaluate(pp_ops.preprocessing(distance_far, input_shape=input_shape, far_set=True), verbose=0)
    no_person_score = model.evaluate(pp_ops.preprocessing(no_person, input_shape=input_shape), verbose=0)
    
    near_f1 = f1(near_score[1], 1-no_person_score[1], 1-near_score[1])
    mid_f1 = f1(mid_score[1], 1-no_person_score[1], 1-mid_score[1])
    far_f1 = f1(far_score[1], 1-no_person_score[1], 1-far_score[1])

    result = pd.DataFrame({
        'distance-near': [near_f1],
        'distance-mid': [mid_f1],
        'distance-far': [far_f1]})
    print(result)

    return result

def miap_eval(model, ds):
    input_shape = model.input_shape[1:]
    
    #Percieved Gender
    miaps_female = ds.filter(lambda x: filter_by_benchmark_set(x, "predominantly_female"))
    miaps_male = ds.filter(lambda x: filter_by_benchmark_set(x, "predominantly_male"))
    miaps_gender_unknown = ds.filter(lambda x: filter_by_benchmark_set(x,"gender_unknown"))

    female_score = model.evaluate(pp_ops.preprocessing(miaps_female, input_shape=input_shape), verbose=0)
    male_score = model.evaluate(pp_ops.preprocessing(miaps_male, input_shape=input_shape), verbose=0)
    gender_unknown_score = model.evaluate(pp_ops.preprocessing(miaps_gender_unknown, input_shape=input_shape), verbose=0)
    
    #Percieved Age
    miaps_young = ds.filter(lambda x: filter_by_benchmark_set(x, "young"))
    miaps_middle = ds.filter(lambda x: filter_by_benchmark_set(x, "middle_age"))
    miaps_older = ds.filter(lambda x: filter_by_benchmark_set(x, "older"))
    miaps_age_unknown = ds.filter(lambda x: filter_by_benchmark_set(x, "age_unknown"))
    
    young_score = model.evaluate(pp_ops.preprocessing(miaps_young, input_shape=input_shape), verbose=0)
    middle_score = model.evaluate(pp_ops.preprocessing(miaps_middle, input_shape=input_shape), verbose=0)
    old_score = model.evaluate(pp_ops.preprocessing(miaps_older, input_shape=input_shape), verbose=0)
    age_unknown_score = model.evaluate(pp_ops.preprocessing(miaps_age_unknown, input_shape=input_shape), verbose=0)
    
    
    no_person = ds.filter(lambda x: not filter_by_benchmark_set(x, "person"))
    no_person_score = model.evaluate(pp_ops.preprocessing(no_person, input_shape=input_shape), verbose=0)
    
    female_f1 = f1(female_score[1], 1-no_person_score[1], 1-female_score[1])
    male_f1 = f1(male_score[1], 1-no_person_score[1], 1-male_score[1])
    gender_unknown_f1 = f1(gender_unknown_score[1], 1-no_person_score[1], 1-gender_unknown_score[1])
    
    young_f1 = f1(young_score[1], 1-no_person_score[1], 1-young_score[1])
    middle_f1 = f1(middle_score[1], 1-no_person_score[1], 1-middle_score[1])
    old_f1 = f1(old_score[1], 1-no_person_score[1], 1-old_score[1])
    age_unknown_f1 = f1(age_unknown_score[1], 1-no_person_score[1], 1-age_unknown_score[1])
    
    result = pd.DataFrame({
        'miap-female': [female_f1],
        'miap-male': [male_f1],
        'miap-unknown-gender': [gender_unknown_f1],
        'miap-young': [young_f1],
        'miap-middle': [middle_f1],
        'miap-old': [old_f1],
        'miap-unknown-age': [age_unknown_f1],
        })
    
    print(result)

    return result


def depiction_eval(model, ds):
    input_shape = model.input_shape[1:]
    
    person = ds.filter(lambda x: filter_by_benchmark_set(x, "person"))
    depictions_person = ds.filter(lambda x: filter_by_benchmark_set(x, "person_depiction"))
    depictions_non_person = ds.filter(lambda x: filter_by_benchmark_set(x, "non-person_depiction"))
    non_depiction_non_person = ds.filter(lambda x: filter_by_benchmark_set(x, "non-person_non-depiction"))
    

    person_score = model.evaluate(pp_ops.preprocessing(person, input_shape=input_shape), verbose=0)
    depictions_persons_score = model.evaluate(pp_ops.preprocessing(depictions_person, input_shape=input_shape), verbose=0)
    depictions_non_persons_score = model.evaluate(pp_ops.preprocessing(depictions_non_person, input_shape=input_shape), verbose=0)
    non_person_no_depictions_score = model.evaluate(pp_ops.preprocessing(non_depiction_non_person, input_shape=input_shape), verbose=0)
    
    depictions_persons_f1 = f1(depictions_persons_score[1], 1.-person_score[1], 1-depictions_persons_score[1])
    depictions_non_persons_f1 = f1(depictions_non_persons_score[1], 1-person_score[1], 1-depictions_non_persons_score[1])
    non_person_no_depictions_f1 = f1(non_person_no_depictions_score[1], 1-person_score[1], 1-non_person_no_depictions_score[1])

    result = pd.DataFrame({
        'depictions_persons': [depictions_persons_f1],
        'depictions_non_persons': [depictions_non_persons_f1],
        'non_person_no_depictions': [non_person_no_depictions_f1],
        })
    
    print(result)

    return result
    
    


def benchmark_suite(model_path, dataset_dir, dataset_type="tfds", evals=["wv", "vww", "distance", "miap", "lighting", "depiction"], model_name=""):
    print("Loading Model:" f"{model_path}")
    model = keras.saving.load_model(model_path)
    
    result = pd.DataFrame({'model_name': [model_name]})
    
    if dataset_type == "hf":
        from datasets import load_dataset
        test = load_dataset("Harvard-Edge/Wake-Vision", split='test', data_dir=dataset_dir)
        test = test.to_tf_dataset()
    elif dataset_type == "tfds":
        import tensorflow_datasets as tfds 
        test = tfds.load(
            "wake_vision",
            data_dir=dataset_dir,
            shuffle_files=False,
            split='test'
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if "wv" in evals:
        input_shape = model.input_shape[1:]
        wv_test = pp_ops.preprocessing(test, input_shape=input_shape)
        wv_test_score = model.evaluate(wv_test, verbose=0)
        print(f"Wake Vision Test Score: {wv_test_score[1]}")
        result = pd.concat([result, pd.DataFrame({"wv_test_score": [wv_test_score[1]]})], axis=1)
        

    if "distance" in evals:
        dist_results = distance_eval(model, test)
        result = pd.concat([result, dist_results], axis=1)
    
    if "miap" in evals:
        miap_results = miap_eval(model, test)
        result = pd.concat([result, miap_results], axis=1)
        
    if "lighting" in evals:
        lighting_results = lighting_eval(model, test)
        result = pd.concat([result, lighting_results], axis=1)
        
    if "depiction" in evals:
        depiction_results = depiction_eval(model, test)
        result = pd.concat([result, depiction_results], axis=1)

    print("Benchmark Complete")
    print(result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default="example_wake_vision_mobilenetv2.keras")
    parser.add_argument("-d", "--dataset_dir", type=str)
    parser.add_argument("-n", "--model_name", type=str, default="")
    parser.add_argument("-e", "--evals", type=str)
    parser.add_argument("-t", "--dataset_type", type=str, default="tfds")
    
    args = parser.parse_args()
    if args.evals:
        evals = args.evals.split(",")
    else:
        evals = ["wv", "vww", "distance", "miap", "lighting", "depiction"]
    
    result = benchmark_suite(
        args.model_path,
        args.dataset_dir,
        dataset_type=args.dataset_type,
        evals=evals,
        model_name=args.model_name)
    
    if args.model_name:
        model_name = args.model_name + "_"
    else:
        model_name = ""
    result.to_csv(f"{model_name}benchmark_results.csv", index=False)