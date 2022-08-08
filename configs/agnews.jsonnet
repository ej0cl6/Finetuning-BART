local dataset = "agnews";
local pretrained_model = "facebook/bart-large";
local data_dir = "./data/rizwan";

local model_name_map = {
    "facebook/bart-base": "bart-base", 
    "facebook/bart-large": "bart-large", 
};

{
    "dataset": dataset, 
	"seed": 0, 
	"gpu_device": 0, 
	"pretrained_model": pretrained_model, 
	"output_dir": "./outputs/%s/%s/" % [dataset, model_name_map[pretrained_model]], 
    "cache_dir": "./cache/", 
    "train_src_file": "%s/%s/train.source" % [data_dir, dataset], 
    "train_tgt_file": "%s/%s/train.target" % [data_dir, dataset], 
    "dev_src_file": "%s/%s/valid.source" % [data_dir, dataset], 
    "dev_tgt_file": "%s/%s/valid.target" % [data_dir, dataset], 
    "test_src_file": "%s/%s/test.source" % [data_dir, dataset], 
    "test_tgt_file": "%s/%s/test.source" % [data_dir, dataset], 
    "max_src_len": 128, 
    "max_tgt_len": 128, 
    "warmup_epoch": 5, 
    "max_epoch": 25, 
    "train_batch_size": 4, 
    "eval_batch_size": 4, 
    "learning_rate": 3e-05, 
    "weight_decay": 1e-05, 
    "grad_clipping": 1.0, 
    "num_beams": 4, 
}
