import os
import shutil
import time
import yaml
import fcntl

def copy_tsv(rank, file_list, clip_bin_list, t5_bin_list, cache_list, target_path, clip_target_path, t5_target_path, copy_t5=True):
    for i, file, clip_bin_file, t5_bin_file, cache_file in zip(range(len(file_list)), file_list, clip_bin_list, t5_bin_list, cache_list):
        target_file = os.path.join(target_path, os.path.basename(file))
        target_clip_bin_file = os.path.join(clip_target_path, os.path.basename(clip_bin_file))
        target_t5_bin_file = os.path.join(t5_target_path, os.path.basename(t5_bin_file))
        target_cache_bin_file = os.path.join(t5_target_path, os.path.basename(cache_file))

        if not os.path.exists(target_file) or os.path.getsize(target_file) != os.path.getsize(file):
            print(f"[{i+1}/{len(file_list)}] rank:{rank} copy {file} to {target_file}")
            shutil.copyfile(file, target_file) # copy tsv
        target_lineidx_file = os.path.splitext(target_file)[0] + '.lineidx'
        lineidx_file = os.path.splitext(file)[0] + '.lineidx'
        if not os.path.exists(target_lineidx_file) or os.path.getsize(target_lineidx_file) != os.path.getsize(lineidx_file):
            shutil.copyfile(os.path.splitext(file)[0] + '.lineidx', os.path.splitext(target_file)[0] + '.lineidx') #copy lineidx

        # handle laion5b format with seperate image tsv and label tsv
        if os.path.exists(file.replace(".tsv", ".label")):
            label_file = file.replace(".tsv", ".label")
            target_label_file = target_file.replace(".tsv", ".label")
            label_linidx_file = label_file + '.lineidx'
            target_label_lineidx_file = target_label_file + '.lineidx'
            if not os.path.exists(target_label_file) or os.path.getsize(target_label_file) != os.path.getsize(label_file):
                shutil.copyfile(label_file, target_label_file) # copy tsv
            if not os.path.exists(target_label_lineidx_file) or os.path.getsize(target_label_lineidx_file) != os.path.getsize(label_linidx_file):
                shutil.copyfile(label_linidx_file, target_label_lineidx_file) #copy lineidx

        if not os.path.exists(target_clip_bin_file) or os.path.getsize(target_clip_bin_file) != os.path.getsize(clip_bin_file):
            shutil.copyfile(clip_bin_file, target_clip_bin_file) # copy clip bin
        clip_bin_txt = clip_bin_file.replace(".bin", ".txt")
        target_clip_bin_txt = target_clip_bin_file.replace(".bin", ".txt")
        if not os.path.exists(target_clip_bin_txt) or os.path.getsize(target_clip_bin_txt) != os.path.getsize(clip_bin_txt):
            shutil.copyfile(clip_bin_txt, target_clip_bin_txt) # copy map.txt

        if copy_t5:
            if not os.path.exists(target_t5_bin_file) or os.path.getsize(target_t5_bin_file) != os.path.getsize(t5_bin_file):
                shutil.copyfile(t5_bin_file, target_t5_bin_file) # copy t5 bin
            t5_bin_txt = t5_bin_file.replace(".bin", ".txt")
            target_t5_bin_txt = target_t5_bin_file.replace(".bin", ".txt")
            if not os.path.exists(target_t5_bin_txt) or os.path.getsize(target_t5_bin_txt) != os.path.getsize(t5_bin_txt):
                shutil.copyfile(t5_bin_txt, target_t5_bin_txt) # copy map.txt

            if not os.path.exists(target_cache_bin_file) or os.path.getsize(target_cache_bin_file) != os.path.getsize(cache_file):
                shutil.copyfile(cache_file, target_cache_bin_file) # copy cache


def write_flag(rank, flag_file):
    with open(flag_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("1")
        fcntl.flock(f, fcntl.LOCK_UN)

def read_flag(rank, world_size, flag_file):
    while True:
        # in case rank 0 has already deleted flag_file.
        if not os.path.exists(flag_file):
            break
        with open(flag_file, 'r') as f:
            content = f.readline()
            if len(content) == world_size:
                break
        time.sleep(5)

def new_yaml_file(yaml_file, rank, tsv_list, target_path, t5_target_path, clip_target_path):
    new_file = os.path.join(target_path, os.path.basename(yaml_file).replace(".yml", f"_{rank}.yml"))
    if os.path.exists(new_file):
        os.remove(new_file)

    with open(yaml_file) as f, open(new_file, 'w') as ywf:
        yaml_docs = yaml.safe_load_all(f)

        for doc in yaml_docs:
            if 'dataset' in doc and doc['dataset']['data_file'] in tsv_list:
                doc['dataset']['data_file'] = os.path.join(target_path, os.path.basename(doc['dataset']['data_file']))
                doc['dataset']['t5_bin'] = os.path.join(t5_target_path, os.path.basename(doc['dataset']['t5_bin']))
                doc['dataset']['clip_bin'] = os.path.join(clip_target_path, os.path.basename(doc['dataset']['clip_bin']))
            yaml.safe_dump(doc, ywf)
            ywf.write("---\n")
    return new_file
