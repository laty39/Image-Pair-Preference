from .lora_dpo import DPOPreferenceRecorder

NODE_CLASS_MAPPINGS = {
    "DPO Preference Recorder": DPOPreferenceRecorder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DPO Preference Recorder": "DPO Preference Recorder"
}

print("------------------------------------------")
print("### Loading: DPO Preference Recorder ###")
print("------------------------------------------")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']