import os
import sys
import subprocess
import platform

# Automatically set the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add the script's directory to the system path
sys.path.append(script_dir)

from sdxl_model_util import finalsave

class simpletuner:

    def __init__(self):
        pass

    def update_dataset_json(self, IMAGE_PATH, RESOLUTION, BASE_DIR):
        import json
        dataset_json_path = os.path.join(BASE_DIR, "dataset.json")
        # Load the current dataset.json content
        with open(dataset_json_path, 'r') as file:
            data = json.load(file)
        # Update the specific entries
        for entry in data:
            if entry['id'] == "something-special-to-remember-by":
                entry['instance_data_dir'] = IMAGE_PATH
                entry['crop_style'] = "center"
                entry['crop_aspect'] = "square"
                entry['resolution_type'] = "pixel"
                entry['resolution'] = RESOLUTION
                entry['minimum_image_size'] = RESOLUTION
                entry['caption_strategy'] = "textfile"
                entry['cache_dir_vae'] = os.path.join(BASE_DIR, "vaecache")
                entry['vae_cache_clear_each_epoch'] = False
            elif entry['id'] == "alt-embed-cache":
                entry['default'] = True
                entry['cache_dir'] = os.path.join(BASE_DIR, "embcache")

        # Remove other entries
        data = [entry for entry in data if entry['id'] in ["something-special-to-remember-by", "alt-embed-cache"]]
        for entry in data:
            if entry['id'] == "something-special-to-remember-by":
                if 'preserve_data_backend_cache' in entry:
                    del entry['preserve_data_backend_cache']
                if 'skip_file_discovery' in entry:
                    del entry['skip_file_discovery']

        # Write the updated content back to the file
        with open(dataset_json_path, 'w') as file:
            json.dump(data, file, indent=4)

        return "dataset.json updated successfully."

    def update_sdxl_env(self, BASE_DIR, **kwargs):
        print("BASE_DIR:", BASE_DIR) 
        env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SimpleTuner", "sdxl-env.sh")
        env_example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SimpleTuner", "sdxl-env.sh.example")
        dataset_json_path = os.path.join(BASE_DIR, "dataset.json")
        dataset_example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SimpleTuner", "multidatabackend.example.json")

        # Check if the env file exists, if not, copy from the example
        if not os.path.exists(env_file_path):
            if os.path.exists(env_example_path):
                import shutil
                shutil.copy(env_example_path, env_file_path)
            else:
                return "sdxl-env.sh.example not found in SimpleTuner directory."

        # Check if dataset.json exists, if not, copy from the example
        if not os.path.exists(dataset_json_path):
            if os.path.exists(dataset_example_path):
                import shutil
                shutil.copy(dataset_example_path, dataset_json_path)
            else:
                return "multidatabackend.example.json not found in SimpleTuner directory."

        # Read the current contents of the file
        with open(env_file_path, 'r') as file:
            lines = file.readlines()

        # Update DATALOADER_CONFIG to point to dataset.json
        kwargs['DATALOADER_CONFIG'] = dataset_json_path
        kwargs['OUTPUT_DIR'] = os.path.join(BASE_DIR, 'savedmodels')
        print("OUTPUT_DIR:", kwargs['OUTPUT_DIR'])

        accelerate_args = "--multi_gpu" if kwargs.get('TRAINING_NUM_PROCESSES', 1) > 1 else ""
        
        # Append additional arguments to existing trainer_extra_args if provided
        trainer_extra_args = kwargs.get('TRAINER_EXTRA_ARGS', "")
        if kwargs.get('TRAIN_TEXT_ENCODER', 'false') == 'true':
            trainer_extra_args += " --train_text_encoder"
        if 'LORA_RANK' in kwargs and kwargs.get('MODEL_TYPE') != 'full':
            trainer_extra_args += f" --lora_rank {kwargs['LORA_RANK']}"
        if 'TEXT_ENCODER_LR' in kwargs:
            trainer_extra_args += f" --text_encoder_lr {kwargs['TEXT_ENCODER_LR']}"
        trainer_extra_args = trainer_extra_args.strip()
        kwargs['BASE_DIR'] = BASE_DIR
        fixed_values = {
            "MAX_NUM_STEPS": "0",
            "RESOLUTION_TYPE": "pixel",
            "VALIDATION_STEPS": "10000000000000000000",
            "ACCELERATE_EXTRA_ARGS": accelerate_args,
            "OUTPUT_DIR": kwargs['OUTPUT_DIR'],
            "PUSH_CHECKPOINTS": "false",
            "TRAINER_EXTRA_ARGS": trainer_extra_args
        }

        # Merge fixed values with kwargs
        kwargs.update(fixed_values)

        # Remove 
        kwargs.pop('IMAGE_PATH', None)
        kwargs.pop('PYTHON_PATH', None)
        kwargs.pop('LORA_RANK', None)
        kwargs.pop('TRAIN_TEXT_ENCODER', None)
        kwargs.pop('just_click_it_before_run', None)
        kwargs.pop('TEXT_ENCODER_LR', None)


        try:
            # Update the lines based on the provided kwargs
            updated_keys = set()
            for key, value in kwargs.items():
                found = False
                for i, line in enumerate(lines):
                    if line.startswith(f"export {key}="):
                        # 清除注释
                        comment_index = line.find('#')
                        if comment_index != -1:
                            line = line[:comment_index].strip()
                        # 更新值
                        if '"' in line or "'" in line:
                            quote_char = '"' if '"' in line else "'"
                            start_index = line.find(quote_char) + 1
                            end_index = line.rfind(quote_char)
                            lines[i] = line[:start_index] + str(value) + line[end_index:] + "\n"
                        else:
                            lines[i] = f"export {key}={value}\n"
                        found = True
                        updated_keys.add(key)
                        break
                if not found:
                    raise ValueError(f"Key '{key}' not found in the environment file.")

            # Ensure each line has only one newline character
            lines = [line.strip() + '\n' for line in lines]

            # Write the updated contents back to the file
            with open(env_file_path, 'w') as file:
                file.writelines(lines)

            return "sdxl-env.sh updated successfully."
        except ValueError as e:
            return str(e)



    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "MODEL_TYPE": (["lora", "full"],),
                              "STABLE_DIFFUSION_3": (["true", "false"],{"default": "false"}),
                              "CHECKPOINTING_STEPS": ("INT", {"default": 1000, "min": 0, "step": 10}),
                              "LEARNING_RATE": ("FLOAT", {"default": 0.000001, "step": 0.000001}),
                              "TEXT_ENCODER_LR": ("FLOAT", {"default": 0.00001,"step": 0.000001}),
                              "MODEL_NAME": ("STRING", {"multiline": True}),
                              "BASE_DIR": ("STRING", {"multiline": True}),
                              "IMAGE_PATH": ("STRING", {"multiline": True}),
                              "PYTHON_PATH": ("STRING", {"multiline": True}),
                              "USE_XFORMERS": (["true", "false"],),
                              "LORA_RANK": ("INT", {"default": 16, "min": 0, "step": 1}),
                              "TRAIN_TEXT_ENCODER": (["false", "true"],),
                              "RESOLUTION": ("INT", {"default": 1024, "min": 0, "step":1}),
                              "TRAIN_BATCH_SIZE": ("INT", {"default": 1, "min": 1, "step": 1}),
                              "NUM_EPOCHS": ("INT", {"default": 10, "min": 1, "step": 1}),
                              "USE_GRADIENT_CHECKPOINTING": (["true", "false"],),
                              "GRADIENT_ACCUMULATION_STEPS": ("INT", {"default": 4, "min": 1, "step": 1}),
                              "OPTIMIZER": (["adamw_bf16", "adamw", "adamw8bit", "adafactor", "dadaptation"], {"default": "adamw_bf16"}),
                              "MIXED_PRECISION": (["no", "bf16"], {"default": "bf16"}),
                              "PURE_BF16": (["true", "false"], {"default": "true"}),
                              "LR_SCHEDULE": (["linear", "sine", "cosine", "cosine_with_restarts", "polynomial", "constant"], {"default": "constant"}),
                              "LR_WARMUP_STEPS": ("INT", {"default": 1000, "min": 0, "step": 1}),
                              "TRAINING_NUM_PROCESSES": ("INT", {"default": 1, "min": 1, "step": 1}),
                              "TRAINER_EXTRA_ARGS": ("STRING", {"multiline": True}),
                              "just_click_it_before_run": ("INT", {"default": 0, "min": 0, "max": 0xffffffffff}),
                              "run": (["no", "yes"],),
                              
                             }}

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "main"
    CATEGORY = "hhy"

    def main(self, run="no", **kwargs):
        output_log = ""
        if run == "yes":
            repo_url = "https://github.com/bghira/SimpleTuner.git"
            repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SimpleTuner")

            if not os.path.exists(repo_dir):
                process = subprocess.run(["git", "clone", repo_url, repo_dir], capture_output=True, text=True)
                output_log += process.stdout
                if process.stderr:
                    output_log += "\nErrors:\n" + process.stderr
            else:
                output_log += "Repository already exists at " + repo_dir + "\n"

        # Check if MODEL_NAME ends with .safetensor and convert if necessary
        if kwargs['MODEL_NAME'].endswith('.safetensors'):
            input_ckpt_path = kwargs['MODEL_NAME']
            model_output_dir = os.path.join(kwargs['BASE_DIR'], 'savedmodels')
            output_log += "Model conversion in progress...\n"
            finalsave(input_ckpt_path, model_output_dir)
            output_log += "Model converted\n"
            kwargs['MODEL_NAME'] = model_output_dir

        # 调用 update_sdxl_env 方法并收集日志
        update_env_result = self.update_sdxl_env(**kwargs)
        output_log += update_env_result + "\n"

        # 调用 update_dataset_json 方法并收集日志
        update_json_result = self.update_dataset_json(kwargs['IMAGE_PATH'], kwargs['RESOLUTION'], kwargs['BASE_DIR'])
        output_log += update_json_result

        if run == "yes":
            try:
                simpletuner_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SimpleTuner")


                # 执行修改后的 train_sdxl.bat 文件
                if platform.system() == "Windows":
                    # Read the train_sdxl.bat file path
                    bat_file_path = os.path.join(simpletuner_dir, "train_sdxl.bat")

                    # Read the content of train_sdxl.bat file
                    with open(bat_file_path, 'r') as file:
                        bat_content = file.read()

                    # Replace %%python%% placeholder with python_path
                    python_path_win = os.path.join(kwargs['PYTHON_PATH'], "python.exe")
                    output_log += f"\nPython path: {python_path_win}\n"
                    bat_content = bat_content.replace("%%python%%", python_path_win)

                    # Write the modified content back to train_sdxl.bat file
                    with open(bat_file_path, 'w') as file:
                        file.write(bat_content)
                    subprocess.Popen(f"cmd /c start cmd.exe /k \"cd {simpletuner_dir} && train_sdxl.bat\"", shell=True)
                    output_log += "\nNew CMD window created and 'dir' command executed with PYTHON_PATH echoed."
                else:
                    python_path = kwargs['PYTHON_PATH']
                    subprocess.Popen(f"xterm -e 'cd {simpletuner_dir}; export PATH={python_path}:$PATH; ./train_sdxl.sh; read'", shell=True)
                    output_log += f"\nExecuted command in terminal: cd {simpletuner_dir}; export PATH={python_path}:$PATH; ./train_sdxl.sh"
                    output_log += "\nNew terminal window created, navigated to SimpleTuner directory, and train_sdxl.sh executed."
            except FileNotFoundError:
                output_log += f"\nError: Command xterm not found, run like 'sudo apt install xterm' install it."

        return (output_log,)



NODE_CLASS_MAPPINGS = {
    "sd not very simple simpletuner by hhy": simpletuner
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "simpletunerpro": "simpletuner_pro"
}
