
## Dataset Setup
The following datasets should be prepared before training.

**Note that if you only want want to run experiments on one specific dataset, you can focus on the setup for that and skip the rest.**

The data directory tree follows the format:
```
|-- dataset
	|-- cc3m
        |  |-- raw_data
        |  |-- clip_embedding
        |  |-- t5_embedding
	|-- cc12m
        |  |-- raw_data
        |  |-- clip_embedding
        |  |-- t5_embedding
	|-- mscoco
        |  |-- raw_data
        |  |-- clip_embedding
        |  |-- t5_embedding
```

### CC3M
We use [img2dataset](https://github.com/rom1504/img2dataset) package to help download the CC3M dataset, 
see the [installation guidance](https://github.com/rom1504/img2dataset#:~:text=url%2Bcaption%20datasets.-,Install,-pip%20install%20img2dataset)
of the package. Here we list main procedures:
* Download the metadata
  
  Go to https://ai.google.com/research/ConceptualCaptions/download and press download, which is a 500MB tsv file.\
  Add the column names at the top of the file with `sed -i '1s/^/caption\turl\n/' Train_GCC-training.tsv`
* Download the images with [img2dataset](https://github.com/rom1504/img2dataset)
  Run this following command. It will download the cc3m dataset under the `output_folder` without resizing image.
  ```bash
  img2dataset --url_list Train_GCC-training.tsv --input_format "tsv"\
        --url_col "url" --caption_col "caption"\
        --output_folder raw_data --processes_count 16 --thread_count 64 --image_size 256 --resize_mode no
  ```
* Archive the raw data
  
  Run the following script to generate mapping file
  ```bash
  python generate_mapping_file.py --raw_dir cc3m/raw_data --outfile cc3m/cc3m_map.json 
  ```
* Generate CLIP Embedding
  ```bash
  python generate_clip_embedding.py --json_file cc3m/cc3m_map.json --outdir cc3m/clip_embedding --batch_size 512
  ```
* Generate T5 Embedding
  ```bash
  python generate_t5_embedding.py --json_file cc3m/cc3m_map.json --outdir cc3m/t5_embedding --batch_size 32
  ```