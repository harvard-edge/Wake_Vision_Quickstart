
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import pandas as pd
import tensorflow as tf

from datasets import load_dataset

import pp_ops

def filter_by_benchmark_set(ds_entry, benchmark_name):
    return ds_entry[benchmark_name] == 1


def preprocessing(ds_split, input_shape=(224, 224, 3)):
    # Convert values from int8 to float32
    ds_split = ds_split.map(pp_ops.cast_images_to_float32)
     # resize small
    resize_small = lambda ds_entry: pp_ops.resize_small(ds_entry, input_shape)
    ds_split = ds_split.map(resize_small)
    # center crop
    center_crop = lambda ds_entry: pp_ops.center_crop(ds_entry, input_shape)
    ds_split = ds_split.map(center_crop)
    
    # Use the official mobilenet preprocessing to normalize images
    ds_split = ds_split.map(pp_ops.mobilenet_preprocessing_wrapper)
    
    ds_split = ds_split.map(pp_ops.convert_to_tf_tensor)

    # Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
    ds_split = ds_split.map(pp_ops.prepare_supervised)
    
    
    
    
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

    person_dark_score = model.evaluate(preprocessing(lighting_dark_person, input_shape=input_shape), verbose=0)
    person_normal_light_score = model.evaluate(preprocessing(lighting_normal_person, input_shape=input_shape), verbose=0)
    person_bright_score = model.evaluate(preprocessing(lighting_bright_person, input_shape=input_shape), verbose=0)
    
    non_person_dark_score = model.evaluate(preprocessing(lighting_dark_non_person, input_shape=input_shape), verbose=0)
    non_person_normal_light_score = model.evaluate(preprocessing(lighting_normal_non_person, input_shape=input_shape), verbose=0)
    non_person_bright_score = model.evaluate(preprocessing(lighting_bright_non_person, input_shape=input_shape), verbose=0)
    
    dark_f1 = f1(person_dark_score[1], 1-non_person_dark_score[1], 1-person_dark_score[1])
    normal_light_f1 = f1(person_normal_light_score[1], 1-non_person_normal_light_score[1], 1-person_normal_light_score[1])
    bright_f1 = f1(person_bright_score[1], 1-non_person_bright_score[1], 1-person_bright_score[1])
    
    result = pd.DataFrame({
        'lighting-dark': [dark_f1],
        'lighting-normal_light': [normal_light_f1],
        'lighting-bright': [bright_f1]})

    print(result)

    return result

# def distance_eval(model, ds):
#     dist_cfg = model_cfg.copy_and_resolve_references()
#     dist_cfg.MIN_BBOX_SIZE = 0.05 #ensure we always use the same min bbox size

#     distance_ds = get_distance_eval(dist_cfg)

#     near_score = model.evaluate(distance_ds["near"], verbose=0)
#     mid_score = model.evaluate(distance_ds["mid"], verbose=0)
#     far_score = model.evaluate(distance_ds["far"], verbose=0)
#     no_person_score = model.evaluate(distance_ds["no_person"], verbose=0)
    
#     near_f1 = f1(near_score[1], 1-no_person_score[1], 1-near_score[1])
#     mid_f1 = f1(mid_score[1], 1-no_person_score[1], 1-mid_score[1])
#     far_f1 = f1(far_score[1], 1-no_person_score[1], 1-far_score[1])

#     result = pd.DataFrame({
#         'distance-near': [near_f1],
#         'distance-mid': [mid_f1],
#         'distance-far': [far_f1]})
#     print(result)

#     return result

# def miap_eval(model, ds):
#     miap_ds = get_miaps(model_cfg, batch_size=1)

#     female_score = model.evaluate(miap_ds["female"], verbose=0)
#     male_score = model.evaluate(miap_ds["male"], verbose=0)
#     gender_unknown_score = model.evaluate(miap_ds["gender_unknown"], verbose=0)
#     young_score = model.evaluate(miap_ds["young"], verbose=0)
#     middle_score = model.evaluate(miap_ds["middle"], verbose=0)
#     old_score = model.evaluate(miap_ds["older"], verbose=0)
#     age_unknown_score = model.evaluate(miap_ds["age_unknown"], verbose=0)
#     no_person_score = model.evaluate(miap_ds["no_person"], verbose=0)
    
#     female_f1 = f1(female_score[1], 1-no_person_score[1], 1-female_score[1])
#     male_f1 = f1(male_score[1], 1-no_person_score[1], 1-male_score[1])
#     gender_unknown_f1 = f1(gender_unknown_score[1], 1-no_person_score[1], 1-gender_unknown_score[1])
    
#     young_f1 = f1(young_score[1], 1-no_person_score[1], 1-young_score[1])
#     middle_f1 = f1(middle_score[1], 1-no_person_score[1], 1-middle_score[1])
#     old_f1 = f1(old_score[1], 1-no_person_score[1], 1-old_score[1])
#     age_unknown_f1 = f1(age_unknown_score[1], 1-no_person_score[1], 1-age_unknown_score[1])
    
#     result = pd.DataFrame({
#         'miap-female': [female_f1],
#         'miap-male': [male_f1],
#         'miap-unknown-gender': [gender_unknown_f1],
#         'miap-young': [young_f1],
#         'miap-middle': [middle_f1],
#         'miap-old': [old_f1],
#         'miap-unknown-age': [age_unknown_f1],
#         })
    
#     print(result)

#     return result


# def depiction_eval(model, ds):
#     depiction_ds = get_depiction_eval(model_cfg, batch_size=1)

#     person_score = model.evaluate(depiction_ds["person"], verbose=0)
#     depictions_persons_score = model.evaluate(depiction_ds["depictions_persons"], verbose=0)
#     depictions_non_persons_score = model.evaluate(depiction_ds["depictions_non_persons"], verbose=0)
#     non_person_no_depictions_score = model.evaluate(depiction_ds["non_person_no_depictions"], verbose=0)
    
#     depictions_persons_f1 = f1(depictions_persons_score[1], 1.-person_score[1], 1-depictions_persons_score[1])
#     depictions_non_persons_f1 = f1(depictions_non_persons_score[1], 1-person_score[1], 1-depictions_non_persons_score[1])
#     non_person_no_depictions_f1 = f1(non_person_no_depictions_score[1], 1-person_score[1], 1-non_person_no_depictions_score[1])

#     result = pd.DataFrame({
#         'depictions_persons': [depictions_persons_f1],
#         'depictions_non_persons': [depictions_non_persons_f1],
#         'non_person_no_depictions': [non_person_no_depictions_f1],
#         })
    
#     print(result)

#     return result
    
    


def benchmark_suite(model_path, evals=["wv", "vww", "distance", "miap", "lighting", "depiction"], model_name=""):
    print("Loading Model:" f"{model_path}")
    model = keras.saving.load_model(model_path)
    
    result = pd.DataFrame({'model_name': [model_name]})
    
    test = load_dataset("Harvard-Edge/Wake-Vision", split='test', streaming=True)
    test = test.to_tf_dataset(
        batch_size=1,
    )

    if "wv" in evals:
        input_shape = model.input_shape[1:]
        wv_test = preprocessing(test, input_shape=input_shape)
        wv_test_score = model.evaluate(wv_test, verbose=0)
        print(f"Wake Vision Test Score: {wv_test_score[1]}")
        result = pd.concat([result, pd.DataFrame({"wv_test_score": [wv_test_score[1]]})], axis=1)
        

    # if "distance" in evals:
    #     dist_results = distance_eval(model, test)
    #     result = pd.concat([result, dist_results], axis=1)
    
    # if "miap" in evals:
    #     miap_results = miap_eval(model, test)
    #     result = pd.concat([result, miap_results], axis=1)
        
    if "lighting" in evals:
        lighting_results = lighting_eval(model, test)
        result = pd.concat([result, lighting_results], axis=1)
        
    # if "depiction" in evals:
    #     depiction_results = depiction_eval(model, test)
    #     result = pd.concat([result, depiction_results], axis=1)

    print("Benchmark Complete")
    print(result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-n", "--model_name", type=str, default="")
    parser.add_argument("-e", "--evals", type=str)
    
    args = parser.parse_args()
    if args.evals:
        evals = args.evals.split(",")
    else:
        evals = ["wv", "vww", "distance", "miap", "lighting", "depiction"]
    
    benchmark_suite(args.model_path, evals=evals, model_name=args.model_name)
     