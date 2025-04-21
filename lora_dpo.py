import torch
import folder_paths
import os
import json
from datetime import datetime
import comfy.utils
import uuid
import locale
from PIL import Image
import numpy as np
import copy # To store previous tensor data safely

# --- Configuration ---
FEEDBACK_DATA_DIR = os.path.join(folder_paths.get_output_directory(), "..", "dpo_data")
DEFAULT_METADATA_FILENAME = "dpo_preferences.jsonl"
os.makedirs(FEEDBACK_DATA_DIR, exist_ok=True)

# --- Language Detection and String Resources ---
try:
    lang_code, encoding = locale.getdefaultlocale()
    detected_language = 'ko' if lang_code and lang_code.lower().startswith('ko') else 'en'
except Exception:
    detected_language = 'en'

TRANSLATIONS = {
    'en': {
        "NODE_NAME": "Image Pair Preference Logger (DPO)", # Renamed slightly
        "CATEGORY": "Image Feedback (DPO)",
        "PREFERENCE_LABEL": "Preference (for PREVIOUS pair)", # Clarified: applies to previous run
        "PREFERENCE_OPTIONS": ["Prefer Image 0", "Prefer Image 1"],
        "PREFERENCE_VALUES": [1.0, -1.0],
        "LOG_THIS_PAIR_LABEL_ON": "Log PREVIOUS Pair", # Clarified: action on previous pair
        "LOG_THIS_PAIR_LABEL_OFF": "SKIP Logging PREVIOUS Pair",
        "NEW_SESSION_LABEL_ON": "Start New Session (for next pair)", # Clarified: affects next storage/log
        "NEW_SESSION_LABEL_OFF": "Keep Current Session",
        "PROMPT_PLACEHOLDER": "Optional prompt text for the CURRENT pair...", # Input is for current
        "WARN_BATCH_SIZE_NOT_TWO": "Warning: Input batch size is {batch_size}, not 2. Storing first two images for next evaluation.", # Updated message
        "WARN_EMPTY_INPUT": "Warning: Input image tensor is empty. Cannot store for next evaluation.",
        "WARN_BATCH_SIZE_LESS_THAN_TWO": "Warning: Input batch size is {batch_size} (< 2). Cannot store pair for next evaluation.",
        "INFO_NEW_SESSION": "Starting new session for next logging: {session_id}",
        "INFO_IMAGE_SAVED": "Saved image copy for logged pair: {filepath}", # Message when logging previous
        "ERROR_IMAGE_SAVE": "Error saving image copy for logged pair (Index {index}, Session: {session_id}): {error}",
        "INFO_PAIR_LOGGED": "Logged preference for PREVIOUS image pair (Session: {session_id}): {filepath}", # Clarified log message
        "INFO_PAIR_SKIPPED": "Skipped logging for PREVIOUS image pair as requested (Toggle was OFF).", # Clarified skip message
        "INFO_FIRST_RUN": "First run or previous data missing. Storing current images for next evaluation.",
        "ERROR_METADATA_SAVE": "Error saving metadata for logged pair ({filepath}): {error}",
        "IMG_PATH_NOT_SAVED": "NOT_SAVED",
        "IMG_PATH_SAVE_FAILED": "SAVE_FAILED"
    },
    'ko': {
        "NODE_NAME": "이미지 쌍 선호도 로거 (DPO)", # 이름 약간 변경
        "CATEGORY": "이미지 피드백 (DPO)",
        "PREFERENCE_LABEL": "선호도 (이전 쌍에 대해)", # 명확화: 이전 실행 결과에 적용
        "PREFERENCE_OPTIONS": ["이미지 0 선호", "이미지 1 선호"],
        "PREFERENCE_VALUES": [1.0, -1.0],
        "LOG_THIS_PAIR_LABEL_ON": "이전 쌍 기록", # 명확화: 이전 쌍에 대한 작업
        "LOG_THIS_PAIR_LABEL_OFF": "이전 쌍 기록 건너뛰기",
        "NEW_SESSION_LABEL_ON": "새 세션 시작 (다음 쌍부터)", # 명확화: 다음 저장/로그에 영향
        "NEW_SESSION_LABEL_OFF": "현재 세션 유지",
        "PROMPT_PLACEHOLDER": "현재 쌍에 대한 참고용 프롬프트 (선택 사항)...", # 입력은 현재용
        "WARN_BATCH_SIZE_NOT_TWO": "경고: 입력 배치 크기가 {batch_size}입니다 (2가 아님). 다음 평가를 위해 처음 2개 이미지를 저장합니다.", # 메시지 업데이트
        "WARN_EMPTY_INPUT": "경고: 입력 이미지 텐서가 비어 있습니다. 다음 평가를 위해 저장할 수 없습니다.",
        "WARN_BATCH_SIZE_LESS_THAN_TWO": "경고: 입력 배치 크기가 {batch_size} (< 2) 입니다. 다음 평가를 위해 쌍을 저장할 수 없습니다.",
        "INFO_NEW_SESSION": "다음 로깅을 위한 새 세션을 시작합니다: {session_id}",
        "INFO_IMAGE_SAVED": "기록된 쌍의 이미지 사본 저장 완료: {filepath}", # 이전 쌍 기록 시 메시지
        "ERROR_IMAGE_SAVE": "기록된 쌍의 이미지 사본 저장 오류 (인덱스 {index}, 세션: {session_id}): {error}",
        "INFO_PAIR_LOGGED": "이전 이미지 쌍에 대한 선호도 기록됨 (세션: {session_id}): {filepath}", # 로그 메시지 명확화
        "INFO_PAIR_SKIPPED": "요청에 따라 이전 이미지 쌍에 대한 기록을 건너<0xEB><0x9B><0x81>니다 (토글이 OFF 상태였음).", # 건너뛰기 메시지 명확화
        "INFO_FIRST_RUN": "첫 실행 또는 이전 데이터 없음. 다음 평가를 위해 현재 이미지를 저장합니다.",
        "ERROR_METADATA_SAVE": "기록된 쌍의 메타데이터 저장 오류 ({filepath}): {error}",
        "IMG_PATH_NOT_SAVED": "저장_안됨",
        "IMG_PATH_SAVE_FAILED": "저장_실패"
    }
}
STR = TRANSLATIONS[detected_language]

class DPOPreferenceRecorder: 
    _data_dir = FEEDBACK_DATA_DIR
    _default_filename = DEFAULT_METADATA_FILENAME
    current_session_id = None # Tracks the active session ID

    # Variables to store data from the PREVIOUS run
    previous_images_tensor = None
    previous_prompt_text = None
    previous_session_id = None

    @classmethod
    def INPUT_TYPES(cls):
        if len(STR["PREFERENCE_OPTIONS"]) != len(STR["PREFERENCE_VALUES"]):
            raise ValueError("Preference options and values lists must have the same length.")
        cls.preference_mapping = {
            option: value for option, value in zip(STR["PREFERENCE_OPTIONS"], STR["PREFERENCE_VALUES"])
        }

        inputs = {
            "required": {
                # Input for the CURRENT run (will be stored for the NEXT run's logging)
                "image": ("IMAGE",),
                # Widgets reflecting user's decision about the PREVIOUS run's images
                "preference": (STR["PREFERENCE_OPTIONS"], {
                    "default": STR["PREFERENCE_OPTIONS"][0] if STR["PREFERENCE_OPTIONS"] else None,
                    "label": STR["PREFERENCE_LABEL"] # Label clarifies it's for previous pair
                }),
                "log_previous_pair": ("BOOLEAN", { # Renamed widget name for clarity
                    "default": True,
                    "label_on": STR["LOG_THIS_PAIR_LABEL_ON"], # Labels reflect action on previous
                    "label_off": STR["LOG_THIS_PAIR_LABEL_OFF"]
                }),
                "start_new_session": ("BOOLEAN", {
                    "default": False,
                    "label_on": STR["NEW_SESSION_LABEL_ON"], # Labels clarify session scope
                    "label_off": STR["NEW_SESSION_LABEL_OFF"]
                }),
            },
            "optional": {
                 # Input for the CURRENT run (will be stored for the NEXT run's logging)
                 "prompt_text_optional": ("STRING", {
                     "multiline": True,
                     "placeholder": STR["PROMPT_PLACEHOLDER"]
                 }),
            }
        }
        return inputs

    RETURN_TYPES = ()
    FUNCTION = "log_previous_and_store_current" # Function name reflects the new logic
    OUTPUT_NODE = True
    CATEGORY = STR["CATEGORY"]

    def log_previous_and_store_current(self,
                                       # 기본값 없는 인자들 먼저
                                       image: torch.Tensor,
                                       preference: str,
                                       log_previous_pair: bool, # 내부 이름 사용
                                       start_new_session: bool,
                                       # 기본값 있는 인자는 맨 뒤로
                                       prompt_text_optional: str = None):

        now = datetime.now()
        metadata_filepath = os.path.join(self._data_dir, self._default_filename)
        node_prefix = f"[{self.__class__.__name__}]" # For cleaner print statements

        # --- 1. Log data from the PREVIOUS run (if available and requested) ---
        previous_data_exists = self.previous_images_tensor is not None and self.previous_session_id is not None

        if previous_data_exists:
            if log_previous_pair: # 여기서 위젯 값 사용
                print(f"{node_prefix} Logging preference for previous pair (Session: {self.previous_session_id})...")
                # --- Save Image Copies for the PREVIOUS pair ---
                session_data_dir = os.path.join(self._data_dir, self.previous_session_id)
                os.makedirs(session_data_dir, exist_ok=True)
                image_paths = [STR['IMG_PATH_SAVE_FAILED']] * 2

                try:
                    # Use the stored tensor from the previous run
                    if self.previous_images_tensor.shape[0] >= 2:
                        for i in range(2):
                            img_filename = f"image_idx{i}.png"
                            img_filepath = os.path.join(session_data_dir, img_filename)
                            relative_path = os.path.join(self.previous_session_id, img_filename).replace("\\", "/")

                            # Make sure the tensor is on CPU before converting to numpy
                            img_tensor_single = self.previous_images_tensor[i].cpu()
                            img_numpy = (img_tensor_single.numpy() * 255).astype(np.uint8)
                            img_pil = Image.fromarray(img_numpy)
                            img_pil.save(img_filepath, compress_level=4)
                            image_paths[i] = relative_path
                            print(f"{node_prefix} {STR['INFO_IMAGE_SAVED'].format(filepath=img_filepath)}")
                    else:
                         print(f"{node_prefix} Error: Previous image tensor did not contain at least 2 images.")
                         # Keep image_paths as SAVE_FAILED

                except Exception as e:
                    print(f"{node_prefix} {STR['ERROR_IMAGE_SAVE'].format(index='N/A', session_id=self.previous_session_id, error=e)}")
                    # image_paths remains SAVE_FAILED

                # --- Determine Score from CURRENT preference widget ---
                comparison_score = self.preference_mapping.get(preference, STR["PREFERENCE_VALUES"][0])
                if preference not in self.preference_mapping:
                     print(f"{node_prefix} Warning: Invalid preference string '{preference}'. Defaulting score to {comparison_score}.")

                # --- Construct Metadata Entry for PREVIOUS pair ---
                metadata_entry = {
                    "assessment_timestamp": now.isoformat(), # Logged at the time of this (N+1) run
                    "session_id": self.previous_session_id, # Use the session ID stored previously
                    "prompt_text_reference": self.previous_prompt_text if self.previous_prompt_text else "",
                    "preference_choice": preference, # Use CURRENT preference widget value
                    "comparison_score": comparison_score,
                    "image_0_relative_path": image_paths[0],
                    "image_1_relative_path": image_paths[1],
                }

                # --- Save Metadata ---
                try:
                    with open(metadata_filepath, 'a', encoding='utf-8') as f:
                        json.dump(metadata_entry, f, ensure_ascii=False)
                        f.write('\n')
                    print(f"{node_prefix} {STR['INFO_PAIR_LOGGED'].format(session_id=self.previous_session_id, filepath=metadata_filepath)}")
                except Exception as e:
                    print(f"{node_prefix} {STR['ERROR_METADATA_SAVE'].format(filepath=metadata_filepath, error=e)}")

            else:
                # Log skip message for the previous pair
                print(f"{node_prefix} {STR['INFO_PAIR_SKIPPED']} (Session: {self.previous_session_id})")
        else:
            # This is the first run or state was lost
             print(f"{node_prefix} {STR['INFO_FIRST_RUN']}")


        # --- 2. Store data from the CURRENT run for the NEXT evaluation ---
        print(f"{node_prefix} Storing current inputs for next potential logging...")
        current_batch_size = image.shape[0]

        # --- Session ID Management for the CURRENT data ---
        # Decide the session ID that will be associated with the *current* images
        # Use the class attribute for session ID tracking across instances (if ComfyUI re-instantiates)
        if start_new_session or DPOPreferenceRecorder.current_session_id is None:
            DPOPreferenceRecorder.current_session_id = now.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
            print(f"{node_prefix} {STR['INFO_NEW_SESSION'].format(session_id=DPOPreferenceRecorder.current_session_id)}")
        session_id_for_current = DPOPreferenceRecorder.current_session_id

        # --- Validate and Store CURRENT Inputs ---
        if current_batch_size == 0:
            print(f"{node_prefix} {STR['WARN_EMPTY_INPUT']}")
            # Clear previous state as current input is invalid for next step
            self.previous_images_tensor = None
            self.previous_prompt_text = None
            self.previous_session_id = None
        elif current_batch_size < 2:
            print(f"{node_prefix} {STR['WARN_BATCH_SIZE_LESS_THAN_TWO'].format(batch_size=current_batch_size)}")
            # Clear previous state
            self.previous_images_tensor = None
            self.previous_prompt_text = None
            self.previous_session_id = None
        else:
            if current_batch_size > 2:
                print(f"{node_prefix} {STR['WARN_BATCH_SIZE_NOT_TWO'].format(batch_size=current_batch_size)}")

            # Store the first two images, the prompt, and the session ID for the next run
            # Use .clone().detach().cpu() for tensors to avoid potential GPU memory issues and ensure safe storage
            try:
                self.previous_images_tensor = image[0:2].clone().detach().cpu() # Store on CPU
                self.previous_prompt_text = prompt_text_optional
                self.previous_session_id = session_id_for_current
                print(f"{node_prefix} Stored current images (first 2) and prompt for Session ID: {self.previous_session_id}. Ready for next evaluation.")
            except Exception as e:
                print(f"{node_prefix} Error storing current image tensor: {e}")
                self.previous_images_tensor = None
                self.previous_prompt_text = None
                self.previous_session_id = None


        # --- Reset instance variables if storage failed ---
        # This is handled implicitly by the checks above setting them to None

        return ()
